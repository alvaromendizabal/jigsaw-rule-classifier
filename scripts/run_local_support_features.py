"""Bounded cached-feature study with independently recoverable stages."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import signal
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from jigsaw_rules.data import normalize, validate_frame
from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest, environment
from scripts.feature_value_audit import ordered_ids
from scripts.local_support_features import BASIC, NAMES, cross_fitted, unit
from scripts.run_behavioral_features import protocol, stage_read, stage_write, weighted_auc_samples

CONFIG = "configs/local_support_features.json"
PRIVATE = "runs/local_support_features"
PUBLIC = "reports/local_support_features"
NOTEBOOK = "notebooks/12_local_support_feature_investigation.ipynb"
REFERENCE = "qwen_raw"
NEW = (
    "answer_only",
    "frozen_basic",
    "frozen_local",
    "frozen_all",
    "shuffled_all",
    "adapted_all",
)
SOURCES = (
    "scripts/local_support_features.py",
    "scripts/run_local_support_features.py",
    "tests/test_local_support_features.py",
    CONFIG,
    "docs/LOCAL_SUPPORT_FEATURES.md",
)
CONTRASTS = [(v, REFERENCE, v + " vs raw Qwen") for v in NEW]
CONTRASTS += [
    ("frozen_all", "answer_only", "All geometry beyond answer-only readout"),
    ("frozen_all", "frozen_basic", "Local density beyond historical geometry"),
    ("frozen_all", "frozen_local", "Basic geometry beyond local density"),
    ("frozen_all", "shuffled_all", "Explicit support labels versus shuffled labels"),
    ("frozen_all", "adapted_all", "Frozen versus adapted geometry"),
]


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_config(spec):
    expected = {
        "primary": "frozen_all",
        "new_fits": 12,
        "neighbors": 10,
        "crossfit_folds": 3,
        "automatic_gpu_authorization": False,
        "minimum_macro_delta": 0.003,
    }
    if any(spec.get(k) != v for k, v in expected.items()):
        raise ValueError("registered feature comparison changed")
    if spec["classifiers"] != {
        "C": 1.0,
        "solver": "liblinear",
        "max_iter": 2000,
        "seed": 2025,
    }:
        raise ValueError("fixed classifier contract changed")
    if not 0 < spec["max_seconds"] <= 240:
        raise ValueError("invalid scientific time budget")
    if not 20 <= spec["bootstrap_replicates"] <= 1000:
        raise ValueError("invalid bootstrap budget")


def verified_file(path, expected):
    if path.is_symlink() or not path.is_file() or digest(path) != expected:
        raise ValueError("missing/changed input; do not recreate: " + str(path))


def load_inputs(root, spec):
    """Use the existing audit's cached inputs; never call its recovery/cloud path."""
    location = root / "reports/feature_value_audit/audit.json"
    verified_file(location, spec["audit_sha256"])
    prior = json.loads(location.read_text())
    if prior["run_id"] != spec["audit_run_id"]:
        raise ValueError("audit identity differs")
    if prior["identity"]["environment"] != environment():
        raise ValueError("environment differs from verified audit")
    for name, checksum in prior["identity"]["source"].items():
        verified_file(root / name, checksum)
    for name, checksum in spec["builder_sources"].items():
        verified_file(root / name, checksum)
    marker = root / "runs/feature_value_audit" / prior["run_id"] / "complete.json"
    saved = json.loads(marker.read_text())
    if saved["identity"] != prior["identity"]:
        raise ValueError("audit completion identity differs")
    for name, checksum in saved["artifacts"].items():
        if Path(name).name != name:
            raise ValueError("invalid audit artifact name")
        verified_file(location.parent / name, checksum)
    verified_file(root / "data/raw/train.csv", spec["train_sha256"])
    frame = pd.read_csv(root / "data/raw/train.csv")
    validate_frame(frame, train=True)
    plan, cohorts = protocol(frame, spec["query_counts"])
    if [c["query_identity"] for c in cohorts] != [c["query_identity"] for c in prior["cohorts"]]:
        raise ValueError("historical query order changed")
    if spec["reference_files"] != prior["identity"]["config"]["reference_files"]:
        raise ValueError("reference provenance differs from verified audit")
    bundles = []
    for index, fold in enumerate(plan["folds"]):
        pin = spec["reference_files"][index]
        path = root / "runs/feature_value_audit/reference" / pin["name"]
        verified_file(path, pin["sha256"])
        if path.stat().st_size != pin["bytes"]:
            raise ValueError("reference file length differs")
        train, query = pd.DataFrame(fold["training"]), pd.DataFrame(fold["queries"])
        if set(query.columns) != {"body", "rule", "row_id"}:
            raise ValueError("query feature inputs must not include targets")
        if set(train.body.map(normalize)) & set(query.body.map(normalize)):
            raise ValueError("query text present in training")
        selected = train.rule.map(normalize).eq(normalize(fold["rule"])).to_numpy()
        if selected.sum() != spec["expected_same_rule_training"][index]:
            raise ValueError("same-rule support count differs")
        if len(train) != prior["cohorts"][index]["training_pairs"]:
            raise ValueError("reconstructed training vector order contract differs")
        ids = query.row_id.to_numpy()
        values = {}
        with np.load(path, allow_pickle=False) as cache:
            ordered_ids(cache["query_row_ids"], ids)
            for model in ("frozen", "adapted"):
                for cohort, count in (("train", len(train)), ("query", len(query))):
                    for kind, width in (("vectors", spec["dimensions"]), ("scores", 3)):
                        key = f"{cohort}_{model}_{kind}"
                        array = cache[key]
                        if array.shape != (count, width) or not np.isfinite(array).all():
                            raise ValueError("cached array shape/finite contract differs: " + key)
                        if kind == "vectors":
                            unit(array)
                        if kind == "scores" and ((array[:, 0] < 0) | (array[:, 0] > 1)).any():
                            raise ValueError("invalid cached score probability")
                        values[key] = (array[selected] if cohort == "train" else array).copy()
        y = frame.set_index("row_id").loc[ids].rule_violation.to_numpy(int)
        observed = roc_auc_score(y, values["query_adapted_scores"][:, 1])
        expected = next(
            m["auc"]
            for m in prior["metrics"]
            if m["fold"] == index and m["model"] == "qwen_adapted_reference"
        )
        if not np.isclose(observed, expected, atol=1e-10, rtol=0):
            raise ValueError("cached Qwen margin does not reproduce the verified audit")
        # Do not attach evaluation targets to feature/fitting bundles.
        bundles.append(
            {
                "train": train.loc[selected].reset_index(drop=True),
                "query": query,
                "arrays": values,
                "rule": fold["rule"],
            }
        )
    return bundles, frame, prior, cohorts


def array_payload(**values):
    data = io.BytesIO()
    np.savez_compressed(data, **values)
    return data.getvalue()


def prepare_features(bundle, folder, key, spec, log):
    banks, diagnostics = {}, []
    train, query, arrays = bundle["train"], bundle["query"], bundle["arrays"]
    y = train.rule_violation.to_numpy(int)
    for name, source, shuffle in (
        ("frozen", "frozen", None),
        ("shuffled", "frozen", spec["seed"]),
        ("adapted", "adapted", None),
    ):
        path = folder / ("features_" + name)
        ident = {**key, "feature_bank": name}
        if stage_read(path, ident) is None:
            tx, qx, inner = cross_fitted(
                arrays["train_" + source + "_vectors"],
                y,
                train.body,
                arrays["query_" + source + "_vectors"],
                neighbors=spec["neighbors"],
                seed=shuffle,
            )
            stats = {
                "bank": name,
                "reference_rows": len(train),
                "query_rows": len(query),
                "inner_folds": inner,
                "self_text_overlap": 0,
                "positive": int(y.sum()),
                "negative": int((1 - y).sum()),
                "training_only_support_statistics": True,
                "query_feature_summary": [
                    {"feature": f, "mean": float(qx[:, j].mean()), "std": float(qx[:, j].std())}
                    for j, f in enumerate(NAMES)
                ],
            }
            stage_write(
                path,
                {
                    "features.npz": array_payload(training=tx, query=qx),
                    "summary.json": json.dumps(stats, indent=2).encode(),
                },
                ident,
            )
        with np.load(path / "features.npz", allow_pickle=False) as loaded:
            tx, qx = loaded["training"].copy(), loaded["query"].copy()
        if (
            tx.shape != (len(train), len(NAMES))
            or qx.shape != (len(query), len(NAMES))
            or not np.isfinite(tx).all()
            or not np.isfinite(qx).all()
        ):
            raise ValueError("feature checkpoint shape mismatch")
        banks[name] = (tx, qx)
        diagnostics.append(json.loads((path / "summary.json").read_text()))
        log.emit("feature_bank_verified", bank=name, fold=key["fold"])
    return banks, diagnostics


def design(bundle, banks, name):
    tx = bundle["arrays"]["train_adapted_scores"][:, 1:2].astype(float)
    qx = bundle["arrays"]["query_adapted_scores"][:, 1:2].astype(float)
    names = ["answer/log_odds"]
    if name != "answer_only":
        bank = (
            "shuffled"
            if name == "shuffled_all"
            else ("adapted" if name == "adapted_all" else "frozen")
        )
        part = (
            slice(0, len(BASIC))
            if name == "frozen_basic"
            else (slice(len(BASIC), None) if name == "frozen_local" else slice(None))
        )
        tx = np.column_stack([tx, banks[bank][0][:, part]])
        qx = np.column_stack([qx, banks[bank][1][:, part]])
        names.extend(NAMES[part])
    return tx, qx, names


def fit_readout(tx, y, qx, repeats, spec):
    if not np.isfinite(tx).all() or not np.isfinite(qx).all():
        raise ValueError("nonfinite feature inputs")
    keep = np.var(tx, axis=0) > 1e-12
    if not keep.any():
        raise ValueError("no nonconstant training features")
    scaler = StandardScaler().fit(tx[:, keep])
    x = scaler.transform(tx[:, keep])
    v = scaler.transform(qx[:, keep])
    model = LogisticRegression(
        C=spec["C"],
        solver=spec["solver"],
        max_iter=spec["max_iter"],
        random_state=spec["seed"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y, sample_weight=repeats)
    return {
        "score": model.decision_function(v),
        "probability": model.predict_proba(v)[:, 1],
        "coefficients": model.coef_.ravel(),
        "intercept": model.intercept_,
        "keep": keep,
        "mean": scaler.mean_,
        "scale": scaler.scale_,
    }


def score_results(records, bundles, frame, spec):
    groups = sorted({normalize(t) for b in bundles for t in b["query"].body})
    lookup = {g: i for i, g in enumerate(groups)}
    reps = spec["bootstrap_replicates"]
    weights = np.random.default_rng(spec["seed"]).multinomial(
        len(groups),
        np.full(len(groups), 1 / len(groups)),
        size=reps,
    )
    metrics, points, draws, pooled = [], {}, {}, []
    for name in (REFERENCE, *NEW):
        aucs, samples, targets, probs, ranked = [], [], [], [], []
        for fold, bundle in enumerate(bundles):
            ids = bundle["query"].row_id.to_numpy()
            y = frame.set_index("row_id").loc[ids].rule_violation.to_numpy(int)
            rec = records[(fold, name)]
            p, score = rec["probability"], rec["score"]
            auc = float(roc_auc_score(y, score))
            metrics.append(
                {
                    "fold": fold,
                    "policy": bundle["rule"],
                    "variant": name,
                    "auc": auc,
                    "brier": float(brier_score_loss(y, p)),
                    "log_loss": float(log_loss(y, p, labels=[0, 1])),
                }
            )
            w = weights[:, [lookup[normalize(t)] for t in bundle["query"].body]]
            samples.append(weighted_auc_samples(y, score, w))
            aucs.append(auc)
            targets.append(y)
            probs.append(p)
            ranked.append((rankdata(score) - 0.5) / len(score))
        points[name], draws[name] = float(np.mean(aucs)), np.mean(samples, axis=0)
        pooled.append(
            {
                "variant": name,
                "macro_auc": points[name],
                "pooled_auc": float(roc_auc_score(np.concatenate(targets), np.concatenate(probs))),
                "ranked_pooled_auc": float(
                    roc_auc_score(np.concatenate(targets), np.concatenate(ranked))
                ),
            }
        )
    delta = np.asarray([points[a] - points[b] for a, b, _ in CONTRASTS])
    bootstrap = np.column_stack([draws[a] - draws[b] for a, b, _ in CONTRASTS])
    valid = np.isfinite(bootstrap).all(axis=1)
    if valid.sum() < 0.9 * reps:
        raise ValueError("too few valid paired bootstrap draws")
    band = float(np.quantile(np.max(np.abs(bootstrap[valid] - delta), axis=1), 0.95))
    comparisons = [
        {
            "candidate": a,
            "reference": b,
            "comparison": title,
            "delta_auc": float(d),
            "simultaneous_low": float(d - band),
            "simultaneous_high": float(d + band),
        }
        for (a, b, title), d in zip(CONTRASTS, delta, strict=True)
    ]
    return metrics, comparisons, pooled


def decide(metrics, comparisons, pooled, spec):
    requirements = []
    for ref in (REFERENCE, "answer_only", "frozen_basic", "shuffled_all"):
        c = next(
            c for c in comparisons if c["candidate"] == spec["primary"] and c["reference"] == ref
        )
        diffs = []
        for fold in range(2):
            m = {m["variant"]: m["auc"] for m in metrics if m["fold"] == fold}
            diffs.append(float(m[spec["primary"]] - m[ref]))
        passed = (
            c["delta_auc"] >= spec["minimum_macro_delta"]
            and c["simultaneous_low"] > 0
            and min(diffs) >= 0
        )
        requirements.append(
            {"reference": ref, "contrast": c, "per_policy_delta": diffs, "passed": bool(passed)}
        )
    ranks = {m["variant"]: m["ranked_pooled_auc"] for m in pooled}
    passed = all(r["passed"] for r in requirements) and (ranks[spec["primary"]] >= ranks[REFERENCE])
    return {
        "decision": "ELIGIBLE_FOR_NEXT_VALIDATION_ONLY" if passed else "DO_NOT_PROMOTE_PRIMARY",
        "primary_requirements": requirements,
    }


def run_study(root, *, max_new_fits=None):
    started = time.monotonic()
    root = Path(root)
    spec = json.loads((root / CONFIG).read_text())
    validate_config(spec)
    bundles, frame, prior, cohorts = load_inputs(root, spec)
    ident = {
        "source": {p: digest(root / p) for p in SOURCES},
        "config": spec,
        "environment": environment(),
        "audit_run": prior["run_id"],
    }
    run_id = json_hash(ident)[:20]
    work, public = root / PRIVATE / run_id, root / PUBLIC
    work.mkdir(parents=True, exist_ok=True)
    with (
        FileLock(str(root / PRIVATE / "study.lock"), timeout=1),
        threadpool_limits(limits=1),
        Progress(work / "events.jsonl", "local_support_round6", heartbeat_seconds=15) as log,
    ):
        finished = work / "finished.json"
        if finished.exists():
            saved = json.loads(finished.read_text())
            if saved["identity"] != ident:
                raise ValueError("finished source identity differs")
            for name, checksum in saved["public_hashes"].items():
                if Path(name).name != name:
                    raise ValueError("invalid result artifact name")
                verified_file(public / name, checksum)
            for fold in range(2):
                for name in NEW:
                    key = {"run_id": run_id, "fold": fold, "variant": name}
                    if stage_read(work / f"fold_{fold}" / name, key) is None:
                        raise ValueError("completed candidate missing; do not refit")
                for name in ("frozen", "shuffled", "adapted"):
                    key = {"run_id": run_id, "fold": fold, "feature_bank": name}
                    if stage_read(work / f"fold_{fold}/features_{name}", key) is None:
                        raise ValueError("completed feature checkpoint missing")
            atomic_json(root / PRIVATE / "last_invocation.json", {"new_fits": 0, "reused": 12})
            log.emit("completed_study_reused", new_fits=0)
            return json.loads((public / "results.json").read_text())
        records, features, coefficients = {}, [], []
        new_fits = reused = 0
        for fold, bundle in enumerate(bundles):
            key = {"run_id": run_id, "fold": fold}
            folder = work / f"fold_{fold}"
            banks, diagnostics = prepare_features(bundle, folder, key, spec, log)
            features.extend({"fold": fold, **d} for d in diagnostics)
            margin = bundle["arrays"]["query_adapted_scores"][:, 1].astype(float)
            records[(fold, REFERENCE)] = {"score": margin, "probability": expit(margin)}
            for name in NEW:
                if time.monotonic() - started > spec["max_seconds"]:
                    raise TimeoutError("scientific budget reached; checkpoints preserved")
                tx, qx, names = design(bundle, banks, name)
                target, fit_key = folder / name, {**key, "variant": name}
                if stage_read(target, fit_key) is None:
                    if max_new_fits is not None and new_fits >= max_new_fits:
                        raise TimeoutError("authored interruption for recovery test")
                    train = bundle["train"]
                    fitted = fit_readout(
                        tx,
                        train.rule_violation.to_numpy(int),
                        qx,
                        train.repeat.to_numpy(float),
                        spec["classifiers"],
                    )
                    fitted["row_ids"] = bundle["query"].row_id.to_numpy()
                    stage_write(target, {"predictions.npz": array_payload(**fitted)}, fit_key)
                    new_fits += 1
                else:
                    reused += 1
                with np.load(target / "predictions.npz", allow_pickle=False) as saved:
                    ordered_ids(saved["row_ids"], bundle["query"].row_id.to_numpy())
                    score, probability = saved["score"].copy(), saved["probability"].copy()
                    keep, coef = saved["keep"], saved["coefficients"]
                    calc = ((qx[:, keep] - saved["mean"]) / saved["scale"]) @ coef
                    calc += saved["intercept"][0]
                    if (
                        score.shape != (len(qx),)
                        or not np.isfinite(score).all()
                        or not np.allclose(calc, score, atol=1e-10, rtol=1e-10)
                        or not np.allclose(expit(score), probability, atol=1e-12, rtol=0)
                    ):
                        raise ValueError("saved readout prediction parity failed")
                    coefficients.extend(
                        {
                            "fold": fold,
                            "variant": name,
                            "feature": str(f),
                            "coefficient": float(c),
                        }
                        for f, c in zip(np.asarray(names)[keep], coef, strict=True)
                    )
                records[(fold, name)] = {"score": score, "probability": probability}
                log.emit(
                    "candidate_complete",
                    fold=fold,
                    variant=name,
                    completed=new_fits + reused,
                    total=12,
                )
        metrics, comparisons, pooled = score_results(records, bundles, frame, spec)
        result = {
            "schema": 1,
            "status": "LOCAL_SUPPORT_ROUND_COMPLETE",
            "run_id": run_id,
            "identity": ident,
            "new_fits": new_fits,
            "reused_new_fits": reused,
            "cached_qwen_controls": 2,
            "cohorts": cohorts,
            "features": features,
            "metrics": metrics,
            "comparisons": comparisons,
            "pooled_metrics": pooled,
            "coefficients": coefficients,
            **decide(metrics, comparisons, pooled, spec),
            "new_neural_inference": 0,
            "automatic_gpu_authorization": False,
            "kaggle_score": None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "limitations": [
                "Repeatedly inspected two-policy development cohort, not fresh validation.",
                "Primary geometry uses frozen vectors; reference statistics are cross-fitted.",
                "Adapted sensitivity vectors and answer scores are in-sample on training supports.",
                "Cross-fitting support statistics is not out-of-fold encoder training.",
                "Inner reference pools are smaller than the full outer prediction pool.",
                "Primary is fixed; secondary winners do not replace it post hoc.",
                "No contrastive encoder, routing rule, or ensemble blend is trained.",
                "Conditional intervals cover this round only, not historical adaptive choices.",
            ],
        }
        atomic_json(public / "results.json", result)
        atomic_json(
            public / "feature_catalog.json",
            {
                "historical_basic": list(BASIC),
                "new_local_density": list(NAMES[len(BASIC) :]),
                "fixed_features": True,
                "screen": "training-only near-constant removal",
            },
        )
        atomic_json(
            finished,
            {
                "identity": ident,
                "public_hashes": {
                    name: digest(public / name) for name in ("results.json", "feature_catalog.json")
                },
            },
        )
        atomic_json(
            root / PRIVATE / "last_invocation.json", {"new_fits": new_fits, "reused": reused}
        )
        log.emit("result_saved", decision=result["decision"], new_fits=new_fits)
        return result


def figures(result):
    import plotly.graph_objects as go

    charts = []
    metrics = pd.DataFrame(result["metrics"])
    figure = go.Figure()
    for fold, part in metrics.groupby("fold"):
        figure.add_bar(x=part.variant, y=part.auc, name=f"Policy fold {fold}")
    figure.update_layout(title="01 | Direct comparison with cached Qwen", barmode="group")
    charts.append(figure)
    cs = result["comparisons"]
    figure = go.Figure(
        go.Scatter(
            x=[c["delta_auc"] for c in cs],
            y=[c["comparison"] for c in cs],
            mode="markers",
            error_x={"array": [c["simultaneous_high"] - c["delta_auc"] for c in cs]},
        )
    )
    figure.add_vline(x=0)
    figure.update_layout(title="02 | Paired conditional simultaneous uncertainty")
    charts.append(figure)
    rows = [c for c in cs if c["reference"] in ("frozen_basic", "frozen_local")]
    figure = go.Figure(go.Bar(x=[c["comparison"] for c in rows], y=[c["delta_auc"] for c in rows]))
    figure.update_layout(title="03 | Family-removal ablations")
    charts.append(figure)
    rows = [
        c
        for c in cs
        if c["reference"] in ("shuffled_all", "adapted_all", "answer_only")
        and c["candidate"] == "frozen_all"
    ]
    figure = go.Figure(go.Bar(x=[c["comparison"] for c in rows], y=[c["delta_auc"] for c in rows]))
    figure.update_layout(title="04 | Label-control and representation sensitivities")
    charts.append(figure)
    coverage = []
    for bank in result["features"]:
        if bank["bank"] == "frozen":
            coverage.extend({"fold": bank["fold"], **f} for f in bank["inner_folds"])
    figure = go.Figure()
    for cls in ("permitted_references", "violating_references"):
        figure.add_bar(
            x=[f"F{r['fold']} / inner {r['inner_fold']}" for r in coverage],
            y=[r[cls] for r in coverage],
            name=cls,
        )
    figure.update_layout(title="05 | Cross-fitted same-rule support coverage", barmode="group")
    charts.append(figure)
    figure = go.Figure()
    for bank in result["features"]:
        if bank["bank"] == "frozen":
            rows = bank["query_feature_summary"][len(BASIC) :]
            figure.add_bar(
                x=[r["feature"] for r in rows],
                y=[r["mean"] for r in rows],
                name=f"Policy fold {bank['fold']}",
            )
    figure.update_layout(title="06 | Query density features: label-free descriptive means")
    charts.append(figure)
    effects = pd.DataFrame(result["coefficients"])
    figure = go.Figure()
    for fold, rows in effects[effects.variant == "frozen_all"].groupby("fold"):
        figure.add_bar(y=rows.feature, x=rows.coefficient, orientation="h", name=f"Fold {fold}")
    figure.update_layout(title="07 | Standardized fitted effects, not causal importance")
    charts.append(figure)
    figure = go.Figure()
    for field in ("brier", "log_loss"):
        values = metrics.groupby("variant")[field].mean()
        figure.add_bar(x=values.index, y=values.values, name=field)
    figure.update_layout(title="08 | Raw probability quality, lower is better", barmode="group")
    charts.append(figure)
    for figure in charts:
        figure.update_layout(
            template="plotly_white",
            height=620,
            margin={"l": 190, "r": 35, "t": 85, "b": 160},
            xaxis={"automargin": True},
            yaxis={"automargin": True},
        )
    return charts


def write_dashboard(root, result):
    import plotly.io as pio

    parts = [
        "<!doctype html><html><meta charset='utf-8'><title>Local support features</title>",
        "<body><h1>Jigsaw: local support-density evidence</h1>",
        "<p>Exploratory cached-feature experiment. Not a Kaggle score.</p>",
    ]
    for index, fig in enumerate(figures(result)):
        parts.append(pio.to_html(fig, full_html=False, include_plotlyjs=(index == 0)))
    parts.append("</body></html>")
    path = root / PUBLIC / "dashboard.html"
    atomic_bytes(path, "\n".join(parts).encode())
    return path


def execute_notebook(root):
    import nbformat
    from nbclient import NotebookClient

    path = root / NOTEBOOK
    marker = root / PRIVATE / "notebook_execution.json"
    book = nbformat.read(path, as_version=4)
    source = json_hash([(c.cell_type, c.source) for c in book.cells])
    result_hash = digest(root / PUBLIC / "results.json")
    if marker.exists():
        saved = json.loads(marker.read_text())
        if (
            saved["source"] != source
            or saved["result_sha256"] != result_hash
            or saved["notebook_sha256"] != digest(path)
            or saved["dashboard_sha256"] != digest(root / PUBLIC / "dashboard.html")
        ):
            raise ValueError("completed notebook changed; preserve it")
        return {**saved, "status": "NOTEBOOK_REUSED", "cells_reexecuted": 0}
    for cell in book.cells:
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
    try:
        NotebookClient(
            book,
            timeout=60,
            kernel_name="jigsaw-rules",
            resources={"metadata": {"path": str(root)}},
        ).execute()
    except BaseException:
        nbformat.write(book, root / PRIVATE / "interrupted_notebook.ipynb")
        raise
    cells = [c for c in book.cells if c.cell_type == "code"]
    charts = sum(
        "application/vnd.plotly.v1+json" in o.get("data", {}) for c in cells for o in c.outputs
    )
    if len(cells) != 8 or any(c.execution_count is None for c in cells) or charts != 8:
        raise ValueError("notebook execution incomplete")
    atomic_bytes(path, nbformat.writes(book).encode())
    record = {
        "status": "NOTEBOOK_EXECUTED",
        "source": source,
        "code_cells": 8,
        "plotly_charts": 8,
        "result_sha256": result_hash,
        "notebook_sha256": digest(path),
        "dashboard_sha256": digest(root / PUBLIC / "dashboard.html"),
    }
    atomic_json(marker, record)
    return record


def export(root, destination=None):
    names = list(SOURCES) + [NOTEBOOK, PUBLIC + "/results.json", PUBLIC + "/feature_catalog.json"]
    names += [
        PRIVATE + "/" + n
        for n in (
            "launcher_report.json",
            "install.json",
            "tests.xml",
            "notebook_execution.json",
            "replay.json",
            "last_invocation.json",
        )
    ]
    names += ["reports/feature_value_audit/audit.json"]
    data = {n: (root / n).read_bytes() for n in names if (root / n).is_file()}
    data["SHA256SUMS.json"] = json.dumps(
        {n: hashlib.sha256(b).hexdigest() for n, b in data.items()}, indent=2
    ).encode()
    target = destination or Path.home() / "jigsaw_feature_round6_return.zip"
    temp = target.with_suffix(".partial")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, value in data.items():
            z.writestr(name, value)
    os.replace(temp, target)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--notebook-only", action="store_true")
    args = parser.parse_args()
    if args.export_only:
        print("RETURN_FILE:", export(args.root))
        return
    if not hasattr(signal, "setitimer"):
        raise ValueError("POSIX hard timer required")

    def expired(signum, frame):
        raise TimeoutError("hard time cap; completed checkpoints preserved")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 240)
    try:
        if args.notebook_only:
            first, second = execute_notebook(args.root), execute_notebook(args.root)
            atomic_json(
                args.root / PRIVATE / "replay.json",
                {
                    "first": first["status"],
                    "second": second["status"],
                    "cells_reexecuted": second["cells_reexecuted"],
                },
            )
        else:
            run_study(args.root)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    main()
