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
from scripts import run_behavior_support_features as prior_run
from scripts import run_passage_support_features as dependency_base
from scripts import run_reference_consistency_features as dependency_other
from scripts.action_window_features import NAMES, cross_fitted
from scripts.run_behavioral_features import stage_read, stage_write, weighted_auc_samples
from scripts.run_local_support_features import array_payload, fit_readout, verified_file
from scripts.run_matched_support_features import prepare_controls as prepare_legacy_controls
from scripts.run_matched_support_features import read_prediction

CONFIG = "configs/action_window_features.json"
PRIVATE = "runs/action_window_features"
PUBLIC = "reports/action_window_features"
NOTEBOOK = "notebooks/21_action_window_investigation.ipynb"
REFERENCE = "qwen_raw"
CONTROLS = ("qwen_raw", "answer_only", "frozen_basic", "context_evidence", "uniform_all")
NEW = (
    "action_windows",
    "qualification_windows",
    "window_all",
    "displaced_all",
    "whole_comment_all",
    "label_null_all",
)

SOURCES = (
    "scripts/action_window_features.py",
    "scripts/run_action_window_features.py",
    "tests/test_action_window_features.py",
    "configs/action_window_features.json",
    "docs/ACTION_WINDOW_FEATURES.md",
    "scripts/run_behavior_support_features.py",
    "scripts/run_conditioned_geometry_features.py",
    "scripts/run_matched_support_features.py",
    "scripts/local_support_features.py",
    *dependency_base.SOURCES,
    *dependency_other.SOURCES,
)
CONTRASTS = [
    ("action_windows", "context_evidence", "action_windows vs context anchor"),
    ("qualification_windows", "context_evidence", "qualification_windows vs context anchor"),
    ("window_all", "context_evidence", "window_all vs context anchor"),
    ("displaced_all", "context_evidence", "displaced_all vs context anchor"),
    ("whole_comment_all", "context_evidence", "whole_comment_all vs context anchor"),
    ("label_null_all", "context_evidence", "label_null_all vs context anchor"),
    ("window_all", "qwen_raw", "Primary vs qwen_raw"),
    ("window_all", "frozen_basic", "Primary vs frozen_basic"),
    ("window_all", "uniform_all", "Primary vs uniform_all"),
    ("window_all", "displaced_all", "Primary vs displaced_all"),
    ("window_all", "whole_comment_all", "Primary vs whole_comment_all"),
    ("window_all", "label_null_all", "Primary vs label_null_all"),
    ("window_all", "action_windows", "Family removal: qualification_windows"),
    ("window_all", "qualification_windows", "Family removal: action_windows"),
]
PRIMARY_REFS = (
    "qwen_raw",
    "frozen_basic",
    "context_evidence",
    "uniform_all",
    "displaced_all",
    "whole_comment_all",
    "label_null_all",
)
BANK_WIDTHS = {"windows": 48, "displaced": 48, "whole_comment": 48, "label_null": 48}
VARIANTS = (*CONTROLS, *NEW)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_config(spec):
    expected = {
        "primary": "window_all",
        "new_fits": 12,
        "cached_control_readouts": 10,
        "crossfit_folds": 3,
        "minimum_macro_delta": 0.003,
        "automatic_gpu_authorization": False,
        "independent_of_other_new_round": True,
        "hyperparameters": {
            "radius_tokens": 12,
            "max_windows": 6,
            "max_characters": 50000,
            "vocabulary_cap_per_family": 6000,
            "top_references": 5,
        },
    }
    if any(spec.get(key) != value for key, value in expected.items()):
        raise ValueError("registered two-round design differs")
    if not 0 < spec["max_seconds"] <= 240:
        raise ValueError("invalid runtime limit")
    if not 20 <= spec["bootstrap_replicates"] <= 1000:
        raise ValueError("invalid bootstrap budget")


def verify_prior(root, spec):
    # Both completed rounds are verified. Neither companion new round is an input.
    for module, number in ((dependency_other, 12), (dependency_base, 13)):
        path = root / module.PUBLIC / "results.json"
        verified_file(path, spec[f"round{number}_results_sha256"])
        rec = json.loads(path.read_text())
        if rec["run_id"] != spec[f"round{number}_run_id"]:
            raise ValueError("completed Round 12/13 identity differs")
        if rec["identity"]["environment"] != environment():
            raise ValueError("completed Round 12/13 environment differs")
        if rec["identity"]["config"]["classifiers"] != spec["classifiers"]:
            raise ValueError("classifier settings changed")
        for name, checksum in rec["identity"]["source"].items():
            verified_file(root / name, checksum)
        directory = root / module.PRIVATE / rec["run_id"]
        marker = json.loads((directory / "finished.json").read_text())
        if marker["identity"] != rec["identity"]:
            raise ValueError("old completion identity differs")
        for name, checksum in marker["public_hashes"].items():
            if Path(name).name != name:
                raise ValueError("unsafe old artifact name")
            verified_file(path.parent / name, checksum)
        for fold in range(2):
            key = {"run_id": rec["run_id"], "fold": fold}
            folder = directory / f"fold_{fold}"
            if stage_read(folder / "feature_bank", {**key, "stage": "feature_bank"}) is None:
                raise ValueError("old feature bank absent; no recomputation")
            for name in module.NEW:
                if stage_read(folder / name, {**key, "variant": name}) is None:
                    raise ValueError("old predictions absent; no refit")
    return dependency_base.verify_prior(root, spec)


def prepare_controls(prior6, directory, bundles, frame, root, previous):
    records, basics = prepare_legacy_controls(prior6, directory, bundles, frame)
    anchor_blocks = []
    work = root / prior_run.PRIVATE / previous["run_id"]
    for fold, bundle in enumerate(bundles):
        folder = work / f"fold_{fold}"
        banks = {}
        for mode, width in prior_run.BANK_WIDTHS.items():
            with np.load(folder / "feature_bank" / (mode + ".npz"), allow_pickle=False) as data:
                tx, qx = data["training"].copy(), data["query"].copy()
            if tx.shape != (len(bundle["train"]), width):
                raise ValueError("prior bank training shape changed")
            if qx.shape != (len(bundle["query"]), width) or not np.isfinite(tx).all():
                raise ValueError("prior bank query schema changed")
            if not np.isfinite(qx).all():
                raise ValueError("prior query feature nonfinite")
            banks[mode] = tx, qx
        for name in ("context_evidence", "uniform_all"):
            tx, qx, names = prior_run.design(bundle, basics[fold], banks, name)
            rec, _, _ = read_prediction(
                folder / name / "predictions.npz", bundle["query"].row_id.to_numpy(), qx
            )
            ids = bundle["query"].row_id.to_numpy()
            labels = frame.set_index("row_id").loc[ids].rule_violation.to_numpy(int)
            expected = next(
                m["auc"] for m in previous["metrics"] if m["variant"] == name and m["fold"] == fold
            )
            if not np.isclose(roc_auc_score(labels, rec["score"]), expected, atol=1e-12):
                raise ValueError("saved control AUC differs")
            records[(fold, name)] = rec
            if name == "context_evidence":
                anchor_blocks.append({"training": tx, "query": qx, "names": names})
    return records, anchor_blocks


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
    mode_map = {
        "displaced_all": "displaced",
        "whole_comment_all": "whole_comment",
        "label_null_all": "label_null",
    }
    mode = mode_map.get(variant, "windows")
    part = (
        slice(0, 24) if variant == NEW[0] else slice(24, None) if variant == NEW[1] else slice(None)
    )
    return (
        np.column_stack([basic["training"], banks[mode][0][:, part]]),
        np.column_stack([basic["query"], banks[mode][1][:, part]]),
        (*basic["names"], *NAMES[part]),
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
        "round9_identity": previous["identity"],
        "completed_source_reports": {str(n): spec[f"round{n}_results_sha256"] for n in (10, 11)},
        "round8_sha256": spec["round8_results_sha256"],
        "audit_identity": audit["identity"],
    }
    run_id = json_hash(ident)[:20]
    work, public = root / PRIVATE / run_id, root / PUBLIC
    work.mkdir(parents=True, exist_ok=True)
    with (
        FileLock(str(root / PRIVATE / "study.lock"), timeout=1),
        threadpool_limits(limits=1),
        Progress(
            work / "events.jsonl", "action_window_features_round15", heartbeat_seconds=15
        ) as log,
    ):
        marker = work / "finished.json"
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved["identity"] != ident:
                raise ValueError("completed Round 15 identity differs")
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
        records, basics = prepare_controls(prior6, directory, bundles, frame, root, previous)
        log.emit("all_control_designs_verified", cached_readouts=10, new_fits=0)
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
            "status": "ACTION_WINDOW_ROUND_COMPLETE",
            "run_id": run_id,
            "identity": ident,
            "new_fits": new_fits,
            "reused_new_fits": reused,
            "reused_prior_controls": 10,
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
                    "This repeatedly inspected development cohort is not "
                    # Keep the literal split through formatting.
                    "independent confirmation."
                    # Keep the literal split through formatting.
                ),
                (
                    "The unpromoted Round 9 context-evidence anchor was chosen "
                    # Keep the literal split through formatting.
                    "adaptively before these rounds."
                    # Keep the literal split through formatting.
                ),
                (
                    "New reference statistics are group-cross-fitted; inherited "
                    # Keep the literal split through formatting.
                    "adapted answer training margins remain in-sample."
                    # Keep the literal split through formatting.
                ),
                (
                    "No semantic counterpart labels or passage labels are "
                    # Keep the literal split through formatting.
                    "invented; original supplied reference labels only."
                    # Keep the literal split through formatting.
                ),
                (
                    "Simultaneous intervals cover this round, not the entire "
                    # Keep the literal split through formatting.
                    "adaptive project search."
                    # Keep the literal split through formatting.
                ),
                (
                    "Both Rounds 14 and 15 were specified before running either. "
                    # Keep the literal split through formatting.
                    "Round 14 is not an input."
                    # Keep the literal split through formatting.
                ),
                (
                    "Windows surround a fixed set of action and qualification "
                    # Keep the literal split through formatting.
                    "cues with a twelve-token radius. Up to six merged windows "
                    # Keep the literal split through formatting.
                    "retain evenly spaced positions when needed. No-cue comments "
                    # Keep the literal split through formatting.
                    "use the whole comment. Displaced windows match count and "
                    # Keep the literal split through formatting.
                    "token lengths, but not their lexical norms; whole-comment "
                    # Keep the literal split through formatting.
                    "controls and explicit coverage report that limitation. "
                    # Keep the literal split through formatting.
                    "Unselected text is deliberately omitted, not claimed "
                    # Keep the literal split through formatting.
                    "preserved. These are deterministic cues, not learned "
                    # Keep the literal split through formatting.
                    "rationales or true syntactic negation scopes."
                    # Keep the literal split through formatting.
                ),
            ],
        }
        atomic_json(public / "results.json", result)
        atomic_json(
            public / "feature_catalog.json",
            {
                "new_primary_columns": 48,
                "families": ["action_windows", "qualification_windows"],
                "reference_label_source": "explicit same-rule supplied labels only",
                "new_vector_inference": False,
                "crossfit_folds": 3,
                "independent_round": 15,
                "new_neural_encoding": False,
                "anchor": "Round 9 context_evidence (exploratory, unpromoted)",
                "feature_construction": "8 reference-only word TF-IDF vocabularies/transforms",
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

    metrics = pd.DataFrame(result["metrics"])
    charts = []
    fig = go.Figure()
    for policy, rows in metrics.groupby("policy", sort=True):
        fig.add_bar(
            x=rows.variant,
            y=rows.auc,
            name="Advertising" if "advert" in policy.lower() else "Legal advice",
        )
    fig.update_layout(title="01 | Same-query policy AUC", barmode="group")
    charts.append(fig)
    contrasts = result["comparisons"]
    fig = go.Figure(
        go.Scatter(
            x=[v["delta_auc"] for v in contrasts],
            y=[v["comparison"] for v in contrasts],
            mode="markers",
            error_x={"array": [v["simultaneous_high"] - v["delta_auc"] for v in contrasts]},
        )
    )
    fig.add_vline(x=0)
    fig.update_layout(title="02 | Conditional simultaneous uncertainty", height=690)
    charts.append(fig)
    comparisons = [v for v in contrasts if v["reference"] in NEW[3:]]
    fig = go.Figure(
        go.Bar(x=[v["comparison"] for v in comparisons], y=[v["delta_auc"] for v in comparisons])
    )
    fig.add_hline(y=0)
    fig.update_layout(title="03 | Mechanism controls, not just extra columns")
    charts.append(fig)
    rows = [v for v in contrasts if v["comparison"].startswith("Family removal")]
    fig = go.Figure(go.Bar(x=[v["comparison"] for v in rows], y=[v["delta_auc"] for v in rows]))
    fig.add_hline(y=0)
    fig.update_layout(title="04 | Matched family-removal evidence")
    charts.append(fig)
    diagnostic = pd.DataFrame(result["diagnostics"])
    outer = diagnostic[diagnostic.inner_fold == "outer_query"]
    labels = [f"Fold {v.fold} | {v['family']} | {v['mode']}" for _, v in outer.iterrows()]
    fig = go.Figure()
    for field in ("mean_windows",):
        fig.add_bar(x=labels, y=outer[field], name=field)
    fig.update_layout(title="05 | Fixed cue-window coverage", barmode="group")
    charts.append(fig)
    fig = go.Figure()
    for field in ("cue_present_fraction", "zero_lexical_fraction", "displaced_changed_fraction"):
        fig.add_bar(x=labels, y=outer[field], name=field)
    fig.update_layout(title="06 | Missing evidence is visible, not invented", barmode="group")
    charts.append(fig)
    effects = pd.DataFrame(result["coefficients"])
    effects = effects[effects.variant == NEW[2]]
    fig = go.Figure()
    if not effects.empty:
        names = effects.assign(mag=effects.coefficient.abs()).groupby("feature").mag.mean()
        keep = names.nlargest(14).index
        for fold, rows in effects[effects.feature.isin(keep)].groupby("fold"):
            fig.add_bar(x=rows.coefficient, y=rows.feature, orientation="h", name=f"Fold {fold}")
    fig.update_layout(title="07 | Standardized associations, not causal effects", height=650)
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
            margin=dict(l=145, r=35, t=90, b=155),
            xaxis=dict(automargin=True),
            yaxis=dict(automargin=True),
        )
    return charts


def write_dashboard(root, result):
    import html

    import plotly.io as pio

    title = "Localized passage evidence against labeled supports"
    parts = [
        "<!doctype html><html><meta charset='utf-8'><title>"
        + html.escape(title)
        + "</title><body><h1>"
        + html.escape(title)
        + "</h1>",
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
    names += [
        "reports/behavior_support_features/results.json",
        "reports/conditioned_geometry_features/results.json",
        "reports/crossmodal_support_features/results.json",
        "reports/multiprototype_features/results.json",
    ]
    data = {n: (root / n).read_bytes() for n in names if (root / n).is_file()}
    data["SHA256SUMS.json"] = json.dumps(
        {n: hashlib.sha256(b).hexdigest() for n, b in data.items()}, indent=2
    ).encode()
    target = destination or Path.home() / "jigsaw_feature_round15_return.zip"
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
