"""Bounded feature study with verified cached controls."""

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
from scipy.stats import rankdata
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from jigsaw_rules.data import normalize
from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest, environment
from scripts import run_matched_support_features as prior_run
from scripts.conditioned_geometry_features import NAMES, cross_fitted
from scripts.local_support_features import BASIC
from scripts.run_behavioral_features import stage_read, stage_write, weighted_auc_samples
from scripts.run_local_support_features import array_payload, fit_readout, verified_file
from scripts.run_matched_support_features import prepare_controls, read_prediction

CONFIG = "configs/conditioned_geometry_features.json"
PRIVATE = "runs/conditioned_geometry_features"
PUBLIC = "reports/conditioned_geometry_features"
NOTEBOOK = "notebooks/14_conditioned_geometry_investigation.ipynb"
REFERENCE = "qwen_raw"
CONTROLS = ("qwen_raw", "answer_only", "frozen_basic")
NEW = ("centered", "deflated", "whitened", "dual", "random_dual", "label_null_dual")
SOURCES = (
    "scripts/conditioned_geometry_features.py",
    "scripts/run_conditioned_geometry_features.py",
    "tests/test_conditioned_geometry_features.py",
    "configs/conditioned_geometry_features.json",
    "docs/CONDITIONED_GEOMETRY_FEATURES.md",
)
CONTRASTS = [
    ("centered", "frozen_basic", "centered vs basic geometry"),
    ("deflated", "frozen_basic", "deflated vs basic geometry"),
    ("whitened", "frozen_basic", "whitened vs basic geometry"),
    ("dual", "frozen_basic", "dual vs basic geometry"),
    ("random_dual", "frozen_basic", "random_dual vs basic geometry"),
    ("label_null_dual", "frozen_basic", "label_null_dual vs basic geometry"),
    ("dual", "qwen_raw", "Primary vs raw Qwen"),
    ("dual", "answer_only", "Primary vs answer-only"),
    ("dual", "random_dual", "Estimated axes vs spectrum-matched random axes"),
    ("dual", "label_null_dual", "Supplied labels vs permuted labels"),
    ("dual", "deflated", "Whitening beyond deflation"),
    ("dual", "whitened", "Deflation beyond whitening"),
    ("deflated", "centered", "Leading-direction removal beyond centering"),
    ("whitened", "centered", "Covariance correction beyond centering"),
]
PRIMARY_REFS = ("qwen_raw", "answer_only", "frozen_basic", "random_dual", "label_null_dual")
BANK_WIDTHS = {
    "centered": 9,
    "deflated": 9,
    "whitened": 9,
    "dual": 18,
    "random_dual": 18,
    "label_null_dual": 18,
}
VARIANTS = (*CONTROLS, *NEW)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_config(spec):
    expected = {
        "primary": "dual",
        "new_fits": 12,
        "cached_control_readouts": 6,
        "crossfit_folds": 3,
        "minimum_macro_delta": 0.003,
        "automatic_gpu_authorization": False,
        "independent_of_other_new_round": True,
        "hyperparameters": {"rank": 32, "remove": 4, "shrinkage": 0.2, "svd_iterations": 3},
    }
    if any(spec.get(k) != value for k, value in expected.items()):
        raise ValueError("registered independent-round design differs")
    if not 0 < spec["max_seconds"] <= 240:
        raise ValueError("invalid runtime limit")
    if not 20 <= spec["bootstrap_replicates"] <= 1000:
        raise ValueError("invalid bootstrap budget")


def verify_prior(root, spec):
    report = root / prior_run.PUBLIC / "results.json"
    verified_file(report, spec["round7_results_sha256"])
    previous = json.loads(report.read_text())
    if previous["run_id"] != spec["round7_run_id"]:
        raise ValueError("Round 7 identity differs")
    if previous["identity"]["environment"] != environment():
        raise ValueError("environment differs from verified Round 7")
    for name, checksum in previous["identity"]["source"].items():
        verified_file(root / name, checksum)
    if previous["identity"]["config"]["classifiers"] != spec["classifiers"]:
        raise ValueError("fixed classifier differs")
    directory = root / prior_run.PRIVATE / previous["run_id"]
    marker = json.loads((directory / "finished.json").read_text())
    if marker["identity"] != previous["identity"]:
        raise ValueError("Round 7 completion identity differs")
    for name, checksum in marker["public_hashes"].items():
        if Path(name).name != name:
            raise ValueError("unsafe prior public name")
        verified_file(report.parent / name, checksum)
    for fold in range(2):
        key = {"run_id": previous["run_id"], "fold": fold}
        if (
            stage_read(directory / f"fold_{fold}/pair_bank", {**key, "stage": "paired_features"})
            is None
        ):
            raise ValueError("Round 7 pair checkpoint missing; no recompute")
        for name in prior_run.NEW:
            if stage_read(directory / f"fold_{fold}" / name, {**key, "variant": name}) is None:
                raise ValueError("Round 7 prediction checkpoint missing; no refit")
    prior6, dir6, bundles, frame, audit, cohorts = prior_run.verify_prior(
        root, previous["identity"]["config"]
    )
    if cohorts != previous["cohorts"]:
        raise ValueError("query cohort differs")
    return previous, prior6, dir6, bundles, frame, audit, cohorts


def feature_bank(bundle, folder, key, spec):
    target = folder / "feature_bank"
    bankkey = {**key, "stage": "feature_bank"}
    if stage_read(target, bankkey) is None:
        arrays, train, query = bundle["arrays"], bundle["train"], bundle["query"]
        values, stats = cross_fitted(
            arrays["train_frozen_vectors"],
            train.rule_violation.to_numpy(int),
            train.body,
            train.rule,
            arrays["query_frozen_vectors"],
            query.body,
            seed=spec["seed"],
        )
        content = {
            mode + ".npz": array_payload(training=v[0], query=v[1]) for mode, v in values.items()
        }
        content["summary.json"] = json.dumps(stats, indent=2).encode()
        stage_write(target, content, bankkey)
    banks = {}
    for mode, width in BANK_WIDTHS.items():
        with np.load(target / (mode + ".npz"), allow_pickle=False) as z:
            tx, qx = z["training"].copy(), z["query"].copy()
        if (
            tx.shape != (len(bundle["train"]), width)
            or qx.shape != (len(bundle["query"]), width)
            or not np.isfinite(tx).all()
            or not np.isfinite(qx).all()
        ):
            raise ValueError("feature checkpoint schema differs")
        banks[mode] = tx, qx
    return banks, json.loads((target / "summary.json").read_text())


def design(bundle, basic, banks, variant):
    if variant not in NEW:
        raise ValueError("unknown candidate")
    names = NAMES if "dual" in variant else tuple(f"{variant}/{n}" for n in BASIC)
    return (
        np.column_stack(
            [bundle["arrays"]["train_adapted_scores"][:, 1], basic[0], banks[variant][0]]
        ),
        np.column_stack(
            [bundle["arrays"]["query_adapted_scores"][:, 1], basic[1], banks[variant][1]]
        ),
        ("answer/log_odds", *BASIC, *names),
    )


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
    for ref in PRIMARY_REFS:
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
    previous, prior6, directory, bundles, frame, audit, cohorts = verify_prior(root, spec)
    ident = {
        "source": {n: digest(root / n) for n in SOURCES},
        "config": spec,
        "environment": environment(),
        "round7_identity": previous["identity"],
        "audit_identity": audit["identity"],
    }
    run_id = json_hash(ident)[:20]
    work, public = root / PRIVATE / run_id, root / PUBLIC
    work.mkdir(parents=True, exist_ok=True)
    with (
        FileLock(str(root / PRIVATE / "study.lock"), timeout=1),
        threadpool_limits(limits=1),
        Progress(
            work / "events.jsonl", "conditioned_geometry_features_round8", heartbeat_seconds=15
        ) as log,
    ):
        marker = work / "finished.json"
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved["identity"] != ident:
                raise ValueError("completed Round 8 identity differs")
            for n, checksum in saved["public_hashes"].items():
                if Path(n).name != n:
                    raise ValueError("unsafe public artifact name")
                verified_file(public / n, checksum)
            for fold in range(2):
                k = {"run_id": run_id, "fold": fold}
                if (
                    stage_read(work / f"fold_{fold}/feature_bank", {**k, "stage": "feature_bank"})
                    is None
                ):
                    raise ValueError("completed feature bank missing")
                for name in NEW:
                    if stage_read(work / f"fold_{fold}" / name, {**k, "variant": name}) is None:
                        raise ValueError("completed candidate missing")
            atomic_json(root / PRIVATE / "last_invocation.json", {"new_fits": 0, "reused": 12})
            log.emit("completed_study_reused", new_fits=0)
            return json.loads((public / "results.json").read_text())
        records, basics = prepare_controls(prior6, directory, bundles, frame)
        log.emit("all_control_designs_verified", cached_readouts=6, new_fits=0)
        new_fits = reused = 0
        metadata, coefficients = [], []
        for fold, bundle in enumerate(bundles):
            folder = work / f"fold_{fold}"
            key = {"run_id": run_id, "fold": fold}
            banks, stats = feature_bank(bundle, folder, key, spec)
            metadata.extend({"fold": fold, **s} for s in stats)
            log.emit("feature_bank_verified", fold=fold)
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
            "status": "CONDITIONED_GEOMETRY_ROUND_COMPLETE",
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
            "diagnostics": metadata,
            "coefficients": coefficients,
            **decide(metrics, comparisons, pooled, spec),
            "new_neural_inference": 0,
            "automatic_gpu_authorization": False,
            "kaggle_score": None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "limitations": [
                (
                    "Exploratory follow-up on repeatedly inspected development "
                    # Keep narrative chunks separate for formatting.
                    "data, not a Kaggle score."
                ),
                (
                    "Both new rounds were fixed before either result; neither "
                    # Keep narrative chunks separate for formatting.
                    "selects the other round."
                ),
                (
                    "Cross-fitting excludes each held training group from every "
                    # Keep narrative chunks separate for formatting.
                    "fitted reference statistic."
                ),
                (
                    "The retained adapted training answer margin is in-sample on "
                    # Keep narrative chunks separate for formatting.
                    "supplied support labels."
                ),
                "Cross-fitting new features does not make that answer margin out-of-fold.",
                "Inner reference pools are smaller than the outer inference pool.",
                (
                    "Conditional intervals cover this round only, not all adaptive "
                    # Keep narrative chunks separate for formatting.
                    "project choices."
                ),
                (
                    "A screen pass is eligibility for new validation, never "
                    # Keep narrative chunks separate for formatting.
                    "automatic GPU authorization."
                ),
                (
                    "Whitening/deflation can remove useful signal; isotropy is a "
                    # Keep narrative chunks separate for formatting.
                    "diagnostic, not a target."
                ),
                (
                    "Random axes share the estimated spectrum and rank, not equal "
                    # Keep narrative chunks separate for formatting.
                    "query row geometry."
                ),
                (
                    "Low-rank regularized whitening retains residual directions; "
                    # Keep narrative chunks separate for formatting.
                    "it is not full whitening."
                ),
            ],
        }
        atomic_json(public / "results.json", result)
        atomic_json(
            public / "feature_catalog.json",
            {
                "new_primary_columns": 18,
                "families": ["deflation similarity (9)", "whitened similarity (9)"],
                "same_rule_only": True,
                "frozen_vectors_only": True,
                "transform_or_reference_fitted_on_inner_training_only": True,
                "no_new_neural_encoding": True,
                "independent_round": 8,
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
    rows = [
        r
        for r in result["comparisons"]
        if r["candidate"] == "dual" and r["reference"] in ("random_dual", "label_null_dual")
    ]
    fig = go.Figure(go.Bar(x=[r["reference"] for r in rows], y=[r["delta_auc"] for r in rows]))
    fig.add_hline(y=0)
    fig.update_layout(title="03 | Mechanism-specific negative controls")
    charts.append(fig)
    rows = [
        r
        for r in result["comparisons"]
        if r["candidate"] == "dual" and r["reference"] in ("deflated", "whitened")
    ]
    fig = go.Figure(go.Bar(x=[r["comparison"] for r in rows], y=[r["delta_auc"] for r in rows]))
    fig.add_hline(y=0)
    fig.update_layout(title="04 | Full-minus-family ablations")
    charts.append(fig)
    stats = pd.DataFrame(result["diagnostics"])
    outer = stats[stats.inner_fold.eq("outer_query")]
    fig = go.Figure()
    for mode, rows in outer.groupby("mode"):
        fig.add_bar(
            x=[f"Fold {v}" for v in rows.fold], y=rows.transformed_reference_mean_cosine, name=mode
        )
    fig.update_layout(title="05 | Reference anisotropy, not prediction quality", barmode="group")
    charts.append(fig)
    rows = outer[outer["mode"].eq("centered")]
    fig = go.Figure()
    for field in ("top_removed_variance_fraction", "captured_variance_fraction"):
        fig.add_bar(x=[f"Fold {v}" for v in rows.fold], y=rows[field], name=field)
    fig.update_layout(title="06 | Estimated covariance spectrum", barmode="group")
    charts.append(fig)
    coefs = pd.DataFrame(result["coefficients"])
    rows = coefs[coefs.variant.eq("dual") & ~coefs.feature.str.startswith(("basic/", "answer/"))]
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
        (
            "<!doctype html><html><meta "
            # Keep narrative chunks separate for formatting.
            "charset='utf-8'><title>Reference-only geometry "
            # Keep narrative chunks separate for formatting.
            "conditioning</title>"
        ),
        "<body><h1>Jigsaw: Reference-only geometry conditioning</h1>",
        "<p>Exploratory cached-feature experiment. Not a Kaggle score.</p>",
    ]
    for index, fig in enumerate(figures(result)):
        parts.append(pio.to_html(fig, full_html=False, include_plotlyjs=(index == 0)))
    parts.append("</body></html>")
    path = root / PUBLIC / "dashboard.html"
    atomic_bytes(path, "\n".join(parts).encode())
    return path


def execute_notebook(root):
    from tempfile import TemporaryDirectory

    import nbformat
    from jupyter_client import KernelManager
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
    sockets = TemporaryDirectory(prefix="jigsaw-ipc-")
    manager = KernelManager(
        kernel_name="jigsaw-rules", transport="ipc", ip=str(Path(sockets.name) / "kernel")
    )
    client = NotebookClient(
        book,
        km=manager,
        timeout=60,
        startup_timeout=30,
        kernel_name="jigsaw-rules",
        resources={"metadata": {"path": str(root)}},
    )
    try:
        client.execute(cleanup_kc=True)
    except BaseException:
        nbformat.write(book, root / PRIVATE / "interrupted_notebook.ipynb")
        raise
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        sockets.cleanup()
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
    names = list(SOURCES[:5]) + [
        NOTEBOOK,
        PUBLIC + "/results.json",
        PUBLIC + "/feature_catalog.json",
    ]
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
    names += ["reports/matched_support_features/results.json"]
    data = {n: (root / n).read_bytes() for n in names if (root / n).is_file()}
    data["SHA256SUMS.json"] = json.dumps(
        {n: hashlib.sha256(b).hexdigest() for n, b in data.items()}, indent=2
    ).encode()
    target = destination or Path.home() / "jigsaw_feature_round8_return.zip"
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
