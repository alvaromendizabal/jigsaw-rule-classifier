"""Bounded paired-exemplar ablation; reuse verified Round 6 controls and inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from jigsaw_rules.data import normalize
from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest, environment
from scripts import run_local_support_features as prior_run
from scripts.feature_value_audit import ordered_ids
from scripts.local_support_features import BASIC
from scripts.matched_support_features import GLOBAL, MODES, NAMES, cross_fitted
from scripts.run_behavioral_features import stage_read, stage_write, weighted_auc_samples
from scripts.run_local_support_features import array_payload, fit_readout, verified_file

CONFIG = "configs/matched_support_features.json"
PRIVATE = "runs/matched_support_features"
PUBLIC = "reports/matched_support_features"
NOTEBOOK = "notebooks/13_matched_support_feature_investigation.ipynb"
REFERENCE = "qwen_raw"
CONTROLS = (REFERENCE, "answer_only", "frozen_basic")
NEW = (
    "paired_global",
    "paired_local",
    "paired_all",
    "repaired_all",
    "orientation_all",
    "unnormalized_all",
)
VARIANTS = (*CONTROLS, *NEW)
SOURCES = (
    "scripts/matched_support_features.py",
    "scripts/run_matched_support_features.py",
    "tests/test_matched_support_features.py",
    CONFIG,
    "docs/MATCHED_SUPPORT_FEATURES.md",
)
CONTRASTS = [(n, "frozen_basic", n + " vs basic geometry") for n in NEW]
CONTRASTS += [
    ("paired_all", REFERENCE, "Primary vs raw Qwen"),
    ("paired_all", "answer_only", "Primary vs answer-only readout"),
    ("paired_all", "repaired_all", "Similar pairing vs random re-pairing"),
    ("paired_all", "orientation_all", "Label orientation vs randomized orientation"),
    ("paired_all", "unnormalized_all", "Unit directions vs raw pair differences"),
    ("paired_all", "paired_global", "Local contrast beyond global contrast"),
    ("paired_all", "paired_local", "Global contrast beyond local contrast"),
]


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_config(spec):
    expected = {
        "primary": "paired_all",
        "new_fits": 12,
        "cached_control_readouts": 6,
        "crossfit_folds": 3,
        "pair_temperature": 0.1,
        "vote_scale": 5.0,
        "minimum_macro_delta": 0.003,
        "automatic_gpu_authorization": False,
    }
    if any(spec.get(k) != v for k, v in expected.items()):
        raise ValueError("registered paired-feature design differs")
    if not 0 < spec["max_seconds"] <= 240:
        raise ValueError("invalid runtime budget")
    if not 20 <= spec["bootstrap_replicates"] <= 1000:
        raise ValueError("invalid bootstrap budget")


def verify_prior(root, spec):
    report = root / prior_run.PUBLIC / "results.json"
    verified_file(report, spec["round6_results_sha256"])
    previous = json.loads(report.read_text())
    if previous["run_id"] != spec["round6_run_id"]:
        raise ValueError("Round 6 identity differs")
    if previous["identity"]["environment"] != environment():
        raise ValueError("environment differs from the verified Round 6 run")
    for name, checksum in previous["identity"]["source"].items():
        verified_file(root / name, checksum)
    spec6 = previous["identity"]["config"]
    if spec6["classifiers"] != spec["classifiers"]:
        raise ValueError("fixed classifier differs from Round 6")
    directory = root / prior_run.PRIVATE / previous["run_id"]
    marker = json.loads((directory / "finished.json").read_text())
    if marker["identity"] != previous["identity"]:
        raise ValueError("Round 6 completion identity differs")
    for name, checksum in marker["public_hashes"].items():
        if Path(name).name != name:
            raise ValueError("unsafe public artifact name")
        verified_file(report.parent / name, checksum)
    # Verify, never recreate, every finished Round 6 scientific stage.
    for fold in range(2):
        for name in prior_run.NEW:
            key = {"run_id": previous["run_id"], "fold": fold, "variant": name}
            if stage_read(directory / f"fold_{fold}" / name, key) is None:
                raise ValueError("prior model checkpoint missing; no refit permitted")
        for name in ("frozen", "shuffled", "adapted"):
            key = {"run_id": previous["run_id"], "fold": fold, "feature_bank": name}
            if stage_read(directory / f"fold_{fold}/features_{name}", key) is None:
                raise ValueError("prior feature checkpoint missing")
    bundles, frame, audit, cohorts = prior_run.load_inputs(root, spec6)
    if cohorts != previous["cohorts"]:
        raise ValueError("query cohort differs from Round 6")
    return previous, directory, bundles, frame, audit, cohorts


def read_prediction(path, ids, design):
    with np.load(path, allow_pickle=False) as data:
        ordered_ids(data["row_ids"], ids)
        keep = data["keep"]
        if keep.shape != (design.shape[1],) or keep.dtype.kind != "b":
            raise ValueError("saved selection mask invalid")
        coef = data["coefficients"]
        mean, scale = data["mean"], data["scale"]
        if (
            any(a.shape != (int(keep.sum()),) for a in (coef, mean, scale))
            or not all(np.isfinite(a).all() for a in (coef, mean, scale))
            or np.any(scale <= 0)
            or data["intercept"].shape != (1,)
        ):
            raise ValueError("saved readout parameters invalid")
        calc = ((design[:, keep] - mean) / scale) @ coef + data["intercept"][0]
        score, probability = data["score"].copy(), data["probability"].copy()
        if (
            score.shape != (len(ids),)
            or probability.shape != score.shape
            or not np.isfinite(score).all()
            or not np.allclose(calc, score, atol=1e-10, rtol=1e-10)
            or not np.allclose(expit(score), probability, atol=1e-12, rtol=0)
        ):
            raise ValueError("saved prediction/design parity failed")
    return {"score": score, "probability": probability}, coef, keep


def prepare_controls(previous, directory, bundles, frame):
    """Verify all six reference readouts before any new fit or geometry construction."""
    controls, basics = {}, []
    lookup = frame.set_index("row_id")
    for fold, bundle in enumerate(bundles):
        folder = directory / f"fold_{fold}"
        ids = bundle["query"].row_id.to_numpy()
        with np.load(folder / "features_frozen/features.npz", allow_pickle=False) as z:
            tx, qx = z["training"].copy(), z["query"].copy()
        expected = len(prior_run.NAMES)
        if (
            tx.shape != (len(bundle["train"]), expected)
            or qx.shape != (len(ids), expected)
            or not np.isfinite(tx).all()
            or not np.isfinite(qx).all()
        ):
            raise ValueError("prior frozen feature shape differs")
        btx, bqx = tx[:, : len(BASIC)], qx[:, : len(BASIC)]
        basics.append((btx, bqx))
        margin = bundle["arrays"]["query_adapted_scores"][:, 1].astype(float)
        controls[(fold, REFERENCE)] = {"score": margin, "probability": expit(margin)}
        for name in CONTROLS[1:]:
            design = margin[:, None]
            if name == "frozen_basic":
                design = np.column_stack([design, bqx])
            controls[(fold, name)], _, _ = read_prediction(
                folder / name / "predictions.npz", ids, design
            )
        labels = lookup.loc[ids].rule_violation.to_numpy(int)
        for name in CONTROLS:
            auc = roc_auc_score(labels, controls[(fold, name)]["score"])
            old = next(
                m["auc"] for m in previous["metrics"] if m["fold"] == fold and m["variant"] == name
            )
            if not np.isclose(auc, old, atol=1e-10, rtol=0):
                raise ValueError("prior readout AUC did not reproduce")
    return controls, basics


def feature_bank(bundle, folder, key, spec):
    target = folder / "pair_bank"
    bankkey = {**key, "stage": "paired_features"}
    if stage_read(target, bankkey) is None:
        arrays, train = bundle["arrays"], bundle["train"]
        values, stats = cross_fitted(
            arrays["train_frozen_vectors"],
            train.rule_violation.to_numpy(int),
            train.body,
            train.rule,
            arrays["query_frozen_vectors"],
            seed=spec["seed"],
        )
        content = {
            mode + ".npz": array_payload(training=v[0], query=v[1]) for mode, v in values.items()
        }
        content["summary.json"] = json.dumps(stats, indent=2).encode()
        stage_write(target, content, bankkey)
    banks = {}
    for mode in MODES:
        with np.load(target / (mode + ".npz"), allow_pickle=False) as z:
            tx, qx = z["training"].copy(), z["query"].copy()
        if (
            tx.shape != (len(bundle["train"]), len(NAMES))
            or qx.shape != (len(bundle["query"]), len(NAMES))
            or not np.isfinite(tx).all()
            or not np.isfinite(qx).all()
        ):
            raise ValueError("pair feature checkpoint shape differs")
        banks[mode] = (tx, qx)
    return banks, json.loads((target / "summary.json").read_text())


def design(bundle, basic, banks, variant):
    mode = {
        "repaired_all": "repaired",
        "orientation_all": "orientation",
        "unnormalized_all": "unnormalized",
    }.get(variant, "matched")
    part = slice(None)
    if variant == "paired_global":
        part = slice(0, len(GLOBAL))
    elif variant == "paired_local":
        part = slice(len(GLOBAL), None)
    train = np.column_stack(
        [bundle["arrays"]["train_adapted_scores"][:, 1], basic[0], banks[mode][0][:, part]]
    )
    query = np.column_stack(
        [bundle["arrays"]["query_adapted_scores"][:, 1], basic[1], banks[mode][1][:, part]]
    )
    return train, query, ("answer/log_odds", *BASIC, *NAMES[part])


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
    for name in VARIANTS:
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
    for ref in (REFERENCE, "answer_only", "frozen_basic", "repaired_all", "orientation_all"):
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
    root = Path(root)
    started = time.monotonic()
    spec = json.loads((root / CONFIG).read_text())
    validate_config(spec)
    previous, directory, bundles, frame, audit, cohorts = verify_prior(root, spec)
    ident = {
        "source": {n: digest(root / n) for n in SOURCES},
        "config": spec,
        "environment": environment(),
        "round6_identity": previous["identity"],
        "audit_identity": audit["identity"],
    }
    run_id = json_hash(ident)[:20]
    work, public = root / PRIVATE / run_id, root / PUBLIC
    work.mkdir(parents=True, exist_ok=True)
    with (
        FileLock(str(root / PRIVATE / "study.lock"), timeout=1),
        threadpool_limits(limits=1),
        Progress(work / "events.jsonl", "matched_support_round7", heartbeat_seconds=15) as log,
    ):
        marker = work / "finished.json"
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved["identity"] != ident:
                raise ValueError("completed Round 7 identity differs")
            for n, checksum in saved["public_hashes"].items():
                if Path(n).name != n:
                    raise ValueError("unsafe public artifact name")
                verified_file(public / n, checksum)
            for fold in range(2):
                k = {"run_id": run_id, "fold": fold}
                if (
                    stage_read(work / f"fold_{fold}/pair_bank", {**k, "stage": "paired_features"})
                    is None
                ):
                    raise ValueError("completed pair bank missing")
                for name in NEW:
                    if stage_read(work / f"fold_{fold}" / name, {**k, "variant": name}) is None:
                        raise ValueError("completed candidate missing")
            atomic_json(root / PRIVATE / "last_invocation.json", {"new_fits": 0, "reused": 12})
            log.emit("completed_study_reused", new_fits=0)
            return json.loads((public / "results.json").read_text())
        records, basics = prepare_controls(previous, directory, bundles, frame)
        log.emit("all_control_designs_verified", cached_readouts=6, new_fits=0)
        new_fits = reused = 0
        metadata, coefficients = [], []
        for fold, bundle in enumerate(bundles):
            folder = work / f"fold_{fold}"
            key = {"run_id": run_id, "fold": fold}
            banks, stats = feature_bank(bundle, folder, key, spec)
            metadata.extend({"fold": fold, **s} for s in stats)
            log.emit("pair_bank_verified", fold=fold)
            for name in NEW:
                if time.monotonic() - started > spec["max_seconds"]:
                    raise TimeoutError("scientific budget reached; checkpoints retained")
                tx, qx, names = design(bundle, basics[fold], banks, name)
                path, fitkey = folder / name, {**key, "variant": name}
                if stage_read(path, fitkey) is None:
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
                    stage_write(path, {"predictions.npz": array_payload(**fitted)}, fitkey)
                    new_fits += 1
                else:
                    reused += 1
                rec, coef, keep = read_prediction(
                    path / "predictions.npz", bundle["query"].row_id.to_numpy(), qx
                )
                records[(fold, name)] = rec
                coefficients.extend(
                    {"fold": fold, "variant": name, "feature": str(f), "coefficient": float(c)}
                    for f, c in zip(np.asarray(names)[keep], coef, strict=True)
                )
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
            "status": "MATCHED_SUPPORT_ROUND_COMPLETE",
            "run_id": run_id,
            "identity": ident,
            "new_fits": new_fits,
            "reused_new_fits": reused,
            "reused_prior_controls": 6,
            "control_design_parity": True,
            "cohorts": cohorts,
            "metrics": metrics,
            "comparisons": comparisons,
            "pooled_metrics": pooled,
            "pairing": metadata,
            "coefficients": coefficients,
            **decide(metrics, comparisons, pooled, spec),
            "new_neural_inference": 0,
            "automatic_gpu_authorization": False,
            "kaggle_score": None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "limitations": [
                "Exploratory follow-up after multiple adaptive rounds on two policies.",
                "Similarity-based pairing is not verified semantic or causal matching.",
                "Frozen vectors and cross-fitted reference statistics do not train an encoder.",
                "The retained adapted training answer score is in-sample on support labels.",
                "Cross-fitting the pair bank does not remove that answer-score limitation.",
                "Inner reference pools are smaller than the outer inference pool.",
                "Degenerate pairs are excluded and their counts reported.",
                "Random re-pairing retains the selected endpoints, not pair distances.",
                "Conditional intervals cover this round only, not all adaptive choices.",
                "A negative result does not exhaust contrastive representation training.",
            ],
        }
        atomic_json(public / "results.json", result)
        atomic_json(
            public / "feature_catalog.json",
            {
                "global_features": list(GLOBAL),
                "local_features": list(NAMES[len(GLOBAL) :]),
                "new_columns": len(NAMES),
                "reused_basic_columns": len(BASIC),
                "same_rule_only": True,
                "frozen_vectors_only": True,
                "pairing": "maximum-total-cosine one-to-one opposing-class assignment",
            },
        )
        atomic_json(
            marker,
            {
                "identity": ident,
                "public_hashes": {
                    n: digest(public / n) for n in ("results.json", "feature_catalog.json")
                },
            },
        )
        atomic_json(
            root / PRIVATE / "last_invocation.json", {"new_fits": new_fits, "reused": reused}
        )
        log.emit("results_saved", decision=result["decision"], new_fits=new_fits)
        return result


def figures(result):
    import plotly.graph_objects as go

    charts = []
    metrics = pd.DataFrame(result["metrics"])
    fig = go.Figure()
    for rule, rows in metrics.groupby("policy", sort=True):
        fig.add_bar(
            x=rows.variant,
            y=rows.auc,
            name="Advertising" if "advert" in rule.lower() else "Legal advice",
        )
    fig.update_layout(title="01 | Same-query policy AUC", barmode="group")
    charts.append(fig)
    rows = result["comparisons"]
    fig = go.Figure(
        go.Scatter(
            x=[v["delta_auc"] for v in rows],
            y=[v["comparison"] for v in rows],
            mode="markers",
            error_x={"array": [v["simultaneous_high"] - v["delta_auc"] for v in rows]},
        )
    )
    fig.add_vline(x=0)
    fig.update_layout(title="02 | Paired conditional uncertainty", height=720)
    charts.append(fig)
    names = ("repaired_all", "orientation_all", "unnormalized_all")
    rows = [r for r in rows if r["candidate"] == "paired_all" and r["reference"] in names]
    fig = go.Figure(go.Bar(x=[r["reference"] for r in rows], y=[r["delta_auc"] for r in rows]))
    fig.add_hline(y=0)
    fig.update_layout(title="03 | Matching, label orientation, and direction normalization")
    charts.append(fig)
    rows = [
        r
        for r in result["comparisons"]
        if r["candidate"] == "paired_all" and r["reference"] in ("paired_global", "paired_local")
    ]
    fig = go.Figure(go.Bar(x=[r["comparison"] for r in rows], y=[r["delta_auc"] for r in rows]))
    fig.add_hline(y=0)
    fig.update_layout(title="04 | Global and local family-removal ablations")
    charts.append(fig)
    stats = pd.DataFrame(result["pairing"])
    outer = stats[stats.inner_fold.eq("outer_query")]
    fig = go.Figure()
    for mode, rows in outer.groupby("mode"):
        fig.add_bar(x=[f"Fold {v}" for v in rows.fold], y=rows.mean_pair_cosine, name=mode)
    fig.update_layout(
        title="05 | Matched endpoint similarity: diagnostic, not semantic proof", barmode="group"
    )
    charts.append(fig)
    rows = stats[stats["mode"].eq("matched")]
    labels = [f"Fold {r.fold} / {r.inner_fold}" for r in rows.itertuples()]
    fig = go.Figure()
    for name in ("usable_pairs", "unused_reference_rows", "degenerate_pairs_excluded"):
        fig.add_bar(x=labels, y=rows[name], name=name)
    fig.update_layout(title="06 | Reference coverage and excluded pairs", barmode="group")
    charts.append(fig)
    coefs = pd.DataFrame(result["coefficients"])
    rows = coefs[coefs.variant.eq("paired_all") & coefs.feature.str.startswith("pair/")]
    fig = go.Figure()
    for fold, part in rows.groupby("fold"):
        fig.add_bar(y=part.feature, x=part.coefficient, orientation="h", name=f"Fold {fold}")
    fig.update_layout(
        title="07 | Standardized fitted effects; not causal importance", height=650, barmode="group"
    )
    charts.append(fig)
    fig = go.Figure()
    for name in ("brier", "log_loss"):
        values = metrics.groupby("variant")[name].mean()
        fig.add_bar(x=values.index, y=values, name=name)
    fig.update_layout(title="08 | Probability quality: lower is better", barmode="group")
    charts.append(fig)
    for fig in charts:
        fig.update_layout(
            template="plotly_white",
            height=fig.layout.height or 530,
            margin=dict(l=130, r=35, t=90, b=140),
            xaxis=dict(automargin=True),
            yaxis=dict(automargin=True),
        )
    return charts


def write_dashboard(root, result):
    import plotly.io as pio

    parts = [
        "<!doctype html><html><meta charset='utf-8'><title>Matched support features</title>",
        "<body><h1>Jigsaw: matched opposing-exemplar evidence</h1>",
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
    client = NotebookClient(
        book,
        timeout=60,
        startup_timeout=30,
        kernel_name="jigsaw-rules",
        resources={"metadata": {"path": str(root)}},
    )
    try:
        client.execute()
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
        "kernel": "jigsaw-rules",
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
    names += ["reports/local_support_features/results.json"]
    data = {n: (root / n).read_bytes() for n in names if (root / n).is_file()}
    data["SHA256SUMS.json"] = json.dumps(
        {n: hashlib.sha256(b).hexdigest() for n, b in data.items()}, indent=2
    ).encode()
    target = destination or Path.home() / "jigsaw_feature_round7_return.zip"
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
