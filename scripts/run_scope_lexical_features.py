"""Round 5: occurrence-level lexical scope with energy-matched controls."""

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
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock
from scipy import sparse
from threadpoolctl import threadpool_limits

from jigsaw_rules.data import normalize, validate_frame
from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest
from scripts import run_evidence_features as evidence
from scripts import run_policy_features as policy
from scripts.evidence_features import row_norms
from scripts.run_behavioral_features import (
    environment,
    metric_row,
    protocol,
    stage_read,
    stage_write,
)
from scripts.run_relational_features import fit_candidate
from scripts.scope_lexical_features import CHANNELS, ScopedLexicon

CONFIG = "configs/scope_lexical_features.json"
NOTEBOOK = "notebooks/10_scope_lexical_investigation.ipynb"
PUBLIC = "reports/scope_lexical_features"
PRIVATE = "runs/scope_lexical_features"
SOURCE_PATHS = (
    "scripts/scope_lexical_features.py",
    "scripts/run_scope_lexical_features.py",
    "tests/test_scope_lexical_features.py",
    CONFIG,
    *evidence.SOURCE_PATHS,
)
CONTROLS = (*evidence.CONTROLS, "rule_both")
ANCHOR = "rule_both"
NEW = {
    "copy_attribution": ("copy", ("attribution",)),
    "scope_attribution": ("scope", ("attribution",)),
    "copy_negation": ("copy", ("negation",)),
    "scope_negation": ("scope", ("negation",)),
    "copy_both": ("copy", ("attribution", "negation")),
    "scope_both": ("scope", ("attribution", "negation")),
}
VARIANTS = (*CONTROLS, *NEW)
CONTRASTS = [(name, ANCHOR, name + " vs lexical evidence anchor") for name in NEW]
CONTRASTS += [
    ("scope_" + name, "copy_" + name, "Scope versus collapsed: " + name)
    for name in ("attribution", "negation", "both")
]
CONTRASTS += [
    ("scope_both", "scope_negation", "Removal: attribution family"),
    ("scope_both", "scope_attribution", "Removal: negation family"),
    (ANCHOR, "condition_lexical", "Round 4 lexical-weight gain (reused)"),
]


def hashed_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def identity(root, config):
    return {
        "source": {p: digest(root / p) for p in SOURCE_PATHS},
        "config": config,
        "environment": environment(),
        "round4_results_sha256": digest(root / evidence.PUBLIC / "results.json"),
        "train_sha256": digest(root / "data/raw/train.csv"),
    }


def verify_prior(root, config):
    """Verify all completed prior stages; do not reconstruct missing fits."""
    path = root / evidence.PUBLIC / "results.json"
    if digest(path) != config["round4_results_sha256"]:
        raise ValueError("round-4 results changed; preserve them")
    fourth = json.loads(path.read_text())
    if fourth["run_id"] != config["round4_run_id"]:
        raise ValueError("round-4 run identity mismatch")
    if fourth["identity"]["environment"] != environment():
        raise ValueError("environment changed since round 4")
    for name, sha in fourth["identity"]["source"].items():
        if digest(root / name) != sha:
            raise ValueError("round-4 source changed: " + name)
    older = evidence.verify_prior(root, fourth["identity"]["config"])
    directory4 = root / evidence.PRIVATE / fourth["run_id"]
    marker = json.loads((directory4 / "finished.json").read_text())
    if marker["identity"] != fourth["identity"]:
        raise ValueError("round-4 finished identity mismatch")
    for name, sha in marker["public_hashes"].items():
        if Path(name).name != name or digest(root / evidence.PUBLIC / name) != sha:
            raise ValueError("round-4 public checksum mismatch")
    for fold in range(2):
        for name in evidence.NEW:
            key = {"run_id": fourth["run_id"], "fold": fold, "variant": name}
            if stage_read(directory4 / f"fold_{fold}" / name, key) is None:
                raise ValueError("round-4 checkpoint missing; do not refit")
    return older, fourth, directory4


def prepare_fold(fold, index, prior, root, config):
    older, fourth, directory4 = prior
    data = evidence.prepare_fold(fold, index, older, root, fourth["identity"]["config"])
    anchor_x, anchor_v, _ = evidence.design(data, ANCHOR)
    p, _, _ = policy.load_prediction(
        directory4 / f"fold_{index}" / ANCHOR / "predictions.npz",
        data["query"].row_id.to_numpy(),
        anchor_v,
    )
    data["refs"][ANCHOR] = {"row_ids": data["query"].row_id.to_numpy(), "probability": p}
    part = data["lexical_parts"]["word"]
    lexicon = ScopedLexicon().fit(
        data["train"], older[0]["identity"]["config"], part["names"], part["transform"]
    )
    x, coverage_x, checks_x = lexicon.transform(data["train"][["body", "rule"]])
    v, coverage_v, checks_v = lexicon.transform(data["query"][["body", "rule"]])
    blocks, checks, coverage = {}, [], []
    for family in CHANNELS:
        blocks[family] = ({}, {})
        for mode in ("scope", "copy"):
            block_x, meta = data["mapper"].augment(
                x[family][mode], data["train"].rule, mode="condition"
            )
            block_v, qmeta = data["mapper"].augment(
                v[family][mode], data["query"].rule, mode="condition"
            )
            if qmeta["unknown_rows"]:
                raise ValueError("query policy missing from supplied training supports")
            blocks[family][0][mode], blocks[family][1][mode] = block_x, block_v
            checks.append({"family": family, "mode": mode, "fold": index, **meta})
        for which in (0, 1):
            a, b = (blocks[family][which][m] for m in ("scope", "copy"))
            if a.shape != b.shape or not np.allclose(row_norms(a), row_norms(b)):
                raise ValueError("policy-scoped block norm or dimension parity failed")
    for dataset, rows, table in (
        ("training", coverage_x, data["train"]),
        ("query", coverage_v, data["query"]),
    ):
        profiles = pd.DataFrame(rows)
        for rule in sorted(set(table.rule)):
            subset = profiles.loc[table.rule.to_numpy() == rule]
            for flag in ("quoted_tokens", "code_tokens", "negation_tokens"):
                coverage.append(
                    {
                        "fold": index,
                        "dataset": dataset,
                        "policy": rule,
                        "flag": flag,
                        "rows_with_flag": int((subset[flag] > 0).sum()),
                        "rows": len(subset),
                        "tokens_with_flag": int(subset[flag].sum()),
                        "tokens": int(subset.tokens.sum()),
                    }
                )
    data.update(
        anchor_x=anchor_x,
        anchor_v=anchor_v,
        scope_blocks=blocks,
        scope_coverage=coverage,
        scope_dimensions=checks,
        scope_energy_checks={"training": checks_x, "query": checks_v},
    )
    return data


def design(data, variant):
    mode, families = NEW[variant]
    x = sparse.hstack(
        [data["anchor_x"]] + [data["scope_blocks"][f][0][mode] for f in families],
        format="csr",
    )
    v = sparse.hstack(
        [data["anchor_v"]] + [data["scope_blocks"][f][1][mode] for f in families],
        format="csr",
    )
    return x, v


def comparisons(records, frame, config):
    from scipy.stats import rankdata
    from sklearn.metrics import roc_auc_score

    from scripts.run_behavioral_features import weighted_auc_samples

    ids = sorted({int(i) for rec in records.values() for i in rec["row_ids"]})
    lookup = frame.set_index("row_id")
    codes, labels = pd.factorize(lookup.loc[ids].body.map(normalize), sort=True)
    group_map = dict(zip(ids, codes, strict=True))
    rng = np.random.default_rng(config["bootstrap_seed"])
    reps = config["bootstrap_replicates"]
    weights = rng.multinomial(len(labels), np.full(len(labels), 1 / len(labels)), size=reps)
    points, samples, pooled = {}, {}, []
    for name in VARIANTS:
        scores, draws, truths, probs, ranks = [], [], [], [], []
        for fold in range(2):
            rec = records[(fold, name)]
            y = lookup.loc[rec["row_ids"]].rule_violation.to_numpy(dtype=int)
            p = rec["probability"]
            scores.append(float(roc_auc_score(y, p)))
            w = weights[:, [group_map[int(i)] for i in rec["row_ids"]]]
            draws.append(weighted_auc_samples(y, p, w))
            truths.append(y)
            probs.append(p)
            ranks.append((rankdata(p, method="average") - 0.5) / len(p))
        points[name] = float(np.mean(scores))
        samples[name] = np.mean(draws, axis=0)
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
    delta = np.asarray([points[a] - points[b] for a, b, _ in CONTRASTS])
    boot = np.column_stack([samples[a] - samples[b] for a, b, _ in CONTRASTS])
    valid = np.isfinite(boot).all(axis=1)
    if valid.sum() < 0.9 * reps:
        raise ValueError("insufficient valid paired bootstrap draws")
    band = float(np.quantile(np.max(np.abs(boot[valid] - delta), axis=1), 0.95))
    out = []
    for (a, b, title), d in zip(CONTRASTS, delta, strict=True):
        out.append(
            {
                "comparison": title,
                "candidate": a,
                "reference": b,
                "delta_auc": float(d),
                "simultaneous_low": float(d - band),
                "simultaneous_high": float(d + band),
                "valid_draws": int(valid.sum()),
            }
        )
    return out, pooled


def decide(metrics, contrasts, pooled, config):
    requirements = []
    primary = config["primary"]
    for ref in (ANCHOR, "copy_both"):
        contrast = next(c for c in contrasts if c["candidate"] == primary and c["reference"] == ref)
        deltas = []
        for fold in range(2):
            m = {row["variant"]: row["auc"] for row in metrics if row["fold"] == fold}
            deltas.append(float(m[primary] - m[ref]))
        passed = (
            contrast["delta_auc"] >= config["minimum_macro_delta"]
            and contrast["simultaneous_low"] > 0
            and min(deltas) >= 0
        )
        requirements.append(
            {
                "reference": ref,
                "contrast": contrast,
                "per_policy_delta": deltas,
                "passed": bool(passed),
            }
        )
    ranks = {p["variant"]: p["ranked_pooled_auc"] for p in pooled}
    delta = ranks[primary] - ranks[ANCHOR]
    passed = all(r["passed"] for r in requirements) and delta >= 0
    return {
        "decision": "ELIGIBLE_FOR_NEXT_VALIDATION_ONLY" if passed else "DO_NOT_PROMOTE_PRIMARY",
        "primary_requirements": requirements,
        "primary_ranked_pooled_delta": float(delta),
    }


def run_study(root: Path, *, max_new_fits=None):
    started = time.monotonic()
    config = json.loads((root / CONFIG).read_text())
    fixed = {
        "primary": "scope_both",
        "new_fits": 12,
        "cached_control_fits": 12,
        "automatic_gpu_authorization": False,
        "negation_window": 4,
    }
    if any(config.get(k) != v for k, v in fixed.items()):
        raise ValueError("registered design differs; preserve the existing experiment")
    if not 0 < config["max_seconds"] <= 240:
        raise ValueError("invalid scientific runtime cap")
    if digest(root / "data/raw/train.csv") != config["train_sha256"]:
        raise ValueError("raw training checksum mismatch")
    prior = verify_prior(root, config)
    frame = pd.read_csv(root / "data/raw/train.csv")
    validate_frame(frame, train=True)
    ident = identity(root, config)
    run_id = hashed_json(ident)[:20]
    work, public = root / PRIVATE / run_id, root / PUBLIC
    work.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    with (
        FileLock(str(root / PRIVATE / "study.lock"), timeout=1),
        threadpool_limits(limits=1),
        Progress(work / "events.jsonl", "scope_lexical_round5", heartbeat_seconds=15) as log,
    ):
        done = work / "finished.json"
        if done.exists():
            complete = json.loads(done.read_text())
            if complete["identity"] != ident:
                raise ValueError("round-5 completion identity differs")
            for name, sha in complete["public_hashes"].items():
                if Path(name).name != name or digest(public / name) != sha:
                    raise ValueError("round-5 public checksum mismatch")
            for fold in range(2):
                for name in NEW:
                    key = {"run_id": run_id, "fold": fold, "variant": name}
                    if stage_read(work / f"fold_{fold}" / name, key) is None:
                        raise ValueError("completed round-5 checkpoint missing")
            atomic_json(work / "last_invocation.json", {"new_fits": 0, "reused_new_fits": 12})
            log.emit("completed_run_reused", new_fits=0, reused_new_fits=12)
            return json.loads((public / "results.json").read_text())
        # Avoid replacing a different completed public study with changed source/config.
        if (public / "results.json").exists():
            previous = json.loads((public / "results.json").read_text())
            if previous.get("run_id") != run_id:
                raise ValueError("different completed round-5 result exists; preserve it")
        plan, cohorts = protocol(frame, config["query_counts"])
        expected = [c["query_identity"] for c in prior[1]["cohorts"]]
        if [c["query_identity"] for c in cohorts] != expected:
            raise ValueError("historical query identity changed")
        atomic_json(work / "identity.json", ident)
        prepared = []
        for index, fold in enumerate(plan["folds"]):
            prepared.append(prepare_fold(fold, index, prior, root, config))
        log.emit("all_control_designs_verified", cached_controls=12, new_fits=0)
        records, metrics, coverage, checks, sizes = {}, [], [], [], []
        new_fits = reused = 0
        for index, data in enumerate(prepared):
            coverage.extend(data["scope_coverage"])
            checks.append({"fold": index, **data["scope_energy_checks"]})
            for name, rec in data["refs"].items():
                records[(index, name)] = rec
            for name in NEW:
                if time.monotonic() - started > config["max_seconds"]:
                    raise TimeoutError("CPU budget exhausted; valid stages preserved")
                folder = work / f"fold_{index}" / name
                key = {"run_id": run_id, "fold": index, "variant": name}
                x, v = design(data, name)
                sizes.append(
                    {
                        "fold": index,
                        "variant": name,
                        "columns": x.shape[1],
                        "nonzeros": x.nnz,
                        "training_rows": x.shape[0],
                    }
                )
                if stage_read(folder, key) is None:
                    if max_new_fits is not None and new_fits >= max_new_fits:
                        raise TimeoutError("authored interruption for recovery test")
                    p, coef, intercept = fit_candidate(
                        x,
                        data["y"],
                        v,
                        data["train"].repeat.to_numpy(float),
                        prior[0][0]["identity"]["config"],
                    )
                    out = io.BytesIO()
                    np.savez_compressed(
                        out,
                        row_ids=data["query"].row_id.to_numpy(),
                        probability=p,
                        coefficients=coef,
                        intercept=intercept,
                    )
                    stage_write(folder, {"predictions.npz": out.getvalue()}, key)
                    new_fits += 1
                else:
                    reused += 1
                p, _, _ = policy.load_prediction(
                    folder / "predictions.npz", data["query"].row_id.to_numpy(), v
                )
                records[(index, name)] = {
                    "row_ids": data["query"].row_id.to_numpy(),
                    "probability": p,
                }
                log.emit(
                    "candidate_complete",
                    completed=new_fits + reused,
                    total=12,
                    fold=index,
                    variant=name,
                    new_fits=new_fits,
                    reused_fits=reused,
                )
        lookup = frame.set_index("row_id")
        for (index, name), rec in records.items():
            labels = lookup.loc[rec["row_ids"]].rule_violation.to_numpy(dtype=int)
            metrics.append(
                metric_row(index, name, rec["probability"], labels, plan["folds"][index]["rule"])
            )
        contrasts, pooled = comparisons(records, frame, config)
        result = {
            "schema": 1,
            "status": "SCOPE_ROUND_COMPLETE",
            "run_id": run_id,
            "identity": ident,
            "round4_run_id": prior[1]["run_id"],
            "new_fits": new_fits,
            "reused_new_fits": reused,
            "reused_prior_controls": 12,
            "candidate_fits": 12,
            "cohorts": cohorts,
            "control_design_parity": True,
            "metrics": metrics,
            "pooled_metrics": pooled,
            "comparisons": contrasts,
            "scope_coverage": coverage,
            "scope_energy_checks": checks,
            "design_sizes": sizes,
            **decide(metrics, contrasts, pooled, config),
            "new_neural_inference": 0,
            "gpu": False,
            "automatic_gpu_authorization": False,
            "kaggle_score": None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "limitations": [
                "Adaptive exploratory follow-up; not an independent holdout.",
                "The Round 4 anchor was not promoted.",
                "Existing training-derived vocabulary, IDF and evidence weights.",
                "Token scope has no access to row targets or external models.",
                "Single-quoted prose and implicit attribution are not resolved.",
                "Four-token negation windows are not syntactic parsing.",
                "A negated recommendation may still violate a rule.",
                "Per-term energy matching does not remove added scope capacity.",
                "Intervals cover 12 contrasts, not the adaptive history.",
                "No accepted-Qwen comparison and no new Kaggle score.",
            ],
        }
        atomic_json(public / "results.json", result)
        atomic_json(
            public / "feature_catalog.json",
            {
                "families": CHANNELS,
                "vocabulary": "same eligible-training word unigrams and bigrams",
                "allocation": "value * sqrt(channel count / total term count)",
                "control": "collapsed term value plus zero channel blocks",
                "weights": "inherited Round 4 per-rule diagnostic weights",
                "no_automatic_label_inversion": True,
                "energy_checks": checks,
            },
        )
        atomic_json(
            done,
            {
                "identity": ident,
                "public_hashes": {
                    n: digest(public / n) for n in ("results.json", "feature_catalog.json")
                },
            },
        )
        atomic_json(
            work / "last_invocation.json",
            {"new_fits": new_fits, "reused_new_fits": reused},
        )
        log.emit("results_saved", decision=result["decision"], new_fits=new_fits)
        return result


def figures(result):
    import plotly.graph_objects as go

    metrics = pd.DataFrame(result["metrics"])
    charts = []
    fig = go.Figure()
    for policy_name, rows in metrics.groupby("policy", sort=True):
        fig.add_bar(
            x=rows.variant,
            y=rows.auc,
            name="Advertising" if "advert" in policy_name.lower() else "Legal advice",
        )
    fig.update_layout(title="01 | Per-policy AUC: scope and cached controls", barmode="group")
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
    fig.update_layout(title="02 | Paired conditional intervals: 12 contrasts", height=650)
    charts.append(fig)
    for prefix, title in (
        ("Scope versus", "03 | Scope information beyond matched feature magnitude"),
        ("Removal:", "04 | Conditional value of each scope family"),
    ):
        chosen = [r for r in rows if r["comparison"].startswith(prefix)]
        fig = go.Figure(
            go.Bar(x=[r["comparison"] for r in chosen], y=[r["delta_auc"] for r in chosen])
        )
        fig.add_hline(y=0)
        fig.update_layout(title=title)
        charts.append(fig)
    fig = go.Figure()
    coverage = pd.DataFrame(result["scope_coverage"])
    for (dataset, fold), rows in coverage.groupby(["dataset", "fold"]):
        counts = rows.groupby("flag")[["rows_with_flag", "rows"]].sum()
        fig.add_bar(
            x=counts.index,
            y=counts.rows_with_flag / counts.rows,
            name=f"{dataset} fold {fold}",
        )
    fig.update_layout(title="05 | Scope prevalence: label-free input diagnostic", barmode="group")
    charts.append(fig)
    fig = go.Figure()
    widths = pd.DataFrame(result["design_sizes"])
    for fold, rows in widths.groupby("fold"):
        fig.add_bar(x=rows.variant, y=rows["columns"], name=f"Fold {fold}")
    fig.update_layout(title="06 | Allocated width, including zero padding", barmode="group")
    charts.append(fig)
    fig = go.Figure()
    for name in ("brier", "log_loss"):
        means = metrics.groupby("variant")[name].mean()
        fig.add_bar(x=means.index, y=means.values, name=name)
    fig.update_layout(title="07 | Probability diagnostics: lower is better", barmode="group")
    charts.append(fig)
    pooled = pd.DataFrame(result["pooled_metrics"])
    fig = go.Figure()
    for name in ("macro_auc", "ranked_pooled_auc"):
        fig.add_bar(x=pooled.variant, y=pooled[name], name=name)
    fig.update_layout(title="08 | Local metric summaries, not a Kaggle score", barmode="group")
    charts.append(fig)
    for fig in charts:
        fig.update_layout(
            template="plotly_white",
            height=fig.layout.height or 540,
            margin=dict(l=130, r=40, t=90, b=170),
            font=dict(size=13),
            xaxis=dict(automargin=True),
            yaxis=dict(automargin=True),
        )
    return charts


def bounded_compute(root: Path):
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "scripts.run_scope_lexical_features",
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
        raise RuntimeError("scope worker stopped; preserve checkpoints and return ZIP")
    return json.loads((root / PUBLIC / "results.json").read_text())


def write_dashboard(root: Path, result):
    import html

    import plotly.io as pio

    title = "Jigsaw | Round 5: occurrence-level lexical scope"
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
    destination = destination or Path.home() / "jigsaw_feature_round5_return.zip"
    names = [
        "scripts/scope_lexical_features.py",
        "scripts/run_scope_lexical_features.py",
        CONFIG,
        "tests/test_scope_lexical_features.py",
        "docs/SCOPE_LEXICAL_FEATURES.md",
        NOTEBOOK,
        PUBLIC + "/results.json",
        PUBLIC + "/feature_catalog.json",
        PRIVATE + "/notebook_execution.json",
        PRIVATE + "/replay.json",
        PRIVATE + "/tests.xml",
        PRIVATE + "/install.json",
        PRIVATE + "/launcher_report.json",
        "reports/behavioral_features/results.json",
        "reports/relational_features/results.json",
        "reports/policy_features/results.json",
        "reports/evidence_features/results.json",
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
