"""Bounded read-only scientific audit and notebook exporter for manual SageMaker use."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import time
import zipfile
from pathlib import Path

import pandas as pd
from filelock import FileLock
from threadpoolctl import threadpool_limits

from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest, environment
from scripts import feature_value_audit as audit

CONFIG, PUBLIC, PRIVATE = audit.CONFIG, audit.PUBLIC, audit.PRIVATE
NOTEBOOK = "notebooks/11_feature_value_audit.ipynb"
NEW_SOURCES = (
    "scripts/feature_value_audit.py",
    "scripts/run_feature_value_audit.py",
    "tests/test_feature_value_audit.py",
    CONFIG,
    "docs/FEATURE_VALUE_AUDIT.md",
)


def ledger(reports):
    selected = [
        (1, "add_behavior", "lexical_control", "Behavior addition"),
        (2, "add_act_roles", "add_behavior", "Actor-action addition"),
        (3, "condition_lexical", "copy_lexical", "Policy conditioning vs copy"),
        (4, "rule_both", "permuted_both", "Diagnostic lexical weighting vs shuffle"),
        (5, "scope_both", "copy_both", "Lexical scope vs copy"),
    ]
    rows = []
    for index, candidate, reference, family in selected:
        result = reports[index][0]
        matches = [
            c
            for c in result["comparisons"]
            if c.get("candidate") == candidate and c.get("reference") == reference
        ]
        if not matches and index in (1, 2):
            label = "add_behavior vs control" if index == 1 else "add_act_roles vs behavior"
            matches = [c for c in result["comparisons"] if c["comparison"] == label]
        if len(matches) != 1:
            raise ValueError("historical matched comparison missing or ambiguous")
        if matches:
            contrast = matches[0]
            rows.append(
                {
                    "round": index,
                    "family": family,
                    **contrast,
                    "historical_decision": result["decision"],
                }
            )
    return rows


def run_audit(root, *, recover=False, home=None):
    started = time.monotonic()
    root = Path(root).resolve()
    config = json.loads((root / CONFIG).read_text())
    work, public = root / PRIVATE, root / PUBLIC
    work.mkdir(parents=True, exist_ok=True)
    with (
        FileLock(str(work / "audit.lock"), timeout=1),
        threadpool_limits(limits=1),
        Progress(work / "events.jsonl", "feature_value_audit", heartbeat_seconds=15) as log,
    ):
        reports = audit.load_rounds(root, config)
        paths, recovery = audit.recover_reference(
            root,
            Path.home() if home is None else Path(home),
            config,
            allow_s3=recover,
            log=log,
        )
        folds, cohorts = audit.load_folds(root, config, reports, paths)
        log.emit("all_inputs_verified", queries=sum(len(f["y"]) for f in folds), new_fits=0)
        ident = {
            "source": {p: digest(root / p) for p in NEW_SOURCES},
            "config": config,
            "environment": environment(),
            "references": [digest(p) for p in paths],
            "reports": {str(k): v[0]["run_id"] for k, v in reports.items()},
        }
        run_id = audit.json_hash(ident)[:20]
        marker = work / run_id / "complete.json"
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved["identity"] != ident:
                raise ValueError("completed audit identity differs")
            for name, checksum in saved["artifacts"].items():
                if Path(name).name != name or digest(public / name) != checksum:
                    raise ValueError("completed audit artifact checksum differs")
            log.emit("completed_audit_reused", new_fits=0, statistics_recomputed=False)
            atomic_json(work / "last_invocation.json", {"status": "AUDIT_REUSED", "fits": 0})
            return json.loads((public / "audit.json").read_text())
        result = audit.summarize(
            folds,
            replicates=config["bootstrap_replicates"],
            seed=config["seed"],
            min_slice_class=config["minimum_slice_class"],
        )
        result.update(
            schema=1,
            status="FEATURE_VALUE_AUDIT_COMPLETE",
            run_id=run_id,
            identity=ident,
            reference_recovery=recovery,
            cohorts=cohorts,
            research_ledger=ledger(reports),
            elapsed_seconds=round(time.monotonic() - started, 3),
            kaggle_score=None,
            limitations=[
                "Cached Qwen scores are from development folds, not hidden Kaggle predictions.",
                "The 881 comments have been repeatedly inspected; this is not fresh validation.",
                "Conditional intervals cover seven contrasts, not all historical selection.",
                "Pairs share comments; pair counts are not independent sample sizes.",
                "Slices are fixed, overlapping, approximate text flags; not causal mechanisms.",
                "Ranking complementarity does not guarantee useful feature additions or blends.",
                "No threshold, calibrator, ensemble weight, or feature is fitted in this audit.",
                "Source/data hashes and exact historical row order are required before scoring.",
            ],
        )
        atomic_json(public / "audit.json", result)
        write_dashboard(root, result)
        atomic_json(
            marker,
            {
                "identity": ident,
                "artifacts": {n: digest(public / n) for n in ("audit.json", "dashboard.html")},
            },
        )
        atomic_json(work / "last_invocation.json", {"status": "AUDIT_COMPUTED", "fits": 0})
        log.emit("audit_saved", run_id=run_id, new_fits=0)
        return result


def figures(result):
    import plotly.graph_objects as go

    table = pd.DataFrame(result["metrics"])
    charts = []
    fig = go.Figure()
    for policy, rows in table.groupby("policy", sort=True):
        label = "Advertising" if "advert" in policy.lower() else "Legal advice"
        fig.add_bar(x=rows.model, y=rows.auc, name=label)
    fig.update_layout(
        title="01 | Same-comment AUC: Qwen and cached feature models", barmode="group"
    )
    charts.append(fig)
    rows = result["contrasts"]
    fig = go.Figure(
        go.Scatter(
            x=[r["delta_auc"] for r in rows],
            y=[r["model"] for r in rows],
            mode="markers",
            error_x={"array": [r["simultaneous_high"] - r["delta_auc"] for r in rows]},
        )
    )
    fig.add_vline(x=0)
    fig.update_layout(title="02 | Conditional AUC differences from Qwen; seven contrasts")
    charts.append(fig)
    pair = pd.DataFrame(result["pair_decomposition"])
    fig = go.Figure()
    grouped = pair.groupby("model", sort=False)
    fig.add_bar(
        x=list(grouped.groups), y=grouped.gained_auc_credit.mean(), name="Ordering improvements"
    )
    fig.add_bar(x=list(grouped.groups), y=-grouped.lost_auc_credit.mean(), name="Ordering damage")
    fig.update_layout(
        title="03 | AUC credit gained versus lost; not independent pairs", barmode="relative"
    )
    charts.append(fig)
    c = pd.DataFrame(result["correlations"])
    p = c.groupby(["left", "right"]).spearman.mean().unstack()
    fig = go.Figure(go.Heatmap(x=p.columns, y=p.index, z=p.to_numpy(), zmin=-1, zmax=1))
    fig.update_layout(title="04 | Mean within-policy rank correlation; no ensemble fitted")
    charts.append(fig)
    slices = pd.DataFrame(result["slices"])
    rows = slices[(slices.present) & (slices.model == audit.REFERENCE)]
    fig = go.Figure()
    for fold, part in rows.groupby("fold"):
        fig.add_bar(x=part.slice, y=part.rows, name=f"Fold {fold}")
    fig.update_layout(title="05 | Fixed context-slice sample counts", barmode="group")
    charts.append(fig)
    rows = slices[(slices.present) & (slices.status == "descriptive_only")]
    fig = go.Figure()
    for (fold, model), part in rows.groupby(["fold", "model"]):
        fig.add_bar(x=part.slice, y=part.auc, name=f"Fold {fold} | {model}")
    fig.update_layout(
        title="06 | Descriptive slice AUC; small-class slices omitted", barmode="group"
    )
    charts.append(fig)
    rows = result.get("research_ledger", [])
    fig = go.Figure(go.Bar(x=[r["family"] for r in rows], y=[r["delta_auc"] for r in rows]))
    fig.add_hline(y=0)
    fig.update_layout(title="07 | Historical matched feature contributions; adaptive studies")
    charts.append(fig)
    fig = go.Figure()
    for field in ("brier", "log_loss"):
        means = table.groupby("model", sort=False)[field].mean()
        fig.add_bar(x=means.index, y=means.values, name=field)
    fig.update_layout(
        title="08 | Raw probability diagnostics, not calibrated rank scores", barmode="group"
    )
    charts.append(fig)
    for fig in charts:
        fig.update_layout(
            template="plotly_white",
            height=600,
            margin={"l": 140, "r": 30, "t": 80, "b": 170},
            font={"size": 12},
            xaxis={"automargin": True},
            yaxis={"automargin": True},
        )
    return charts


def write_dashboard(root, result):
    import plotly.io as pio

    parts = [
        "<!doctype html><html><meta charset='utf-8'><title>Feature-value audit</title>",
        "<body><h1>Jigsaw: feature-value audit</h1>",
        "<p>Exploratory cached comparisons. No new Kaggle score, fitting or ensemble.</p>",
    ]
    for i, chart in enumerate(figures(result)):
        parts.append(pio.to_html(chart, full_html=False, include_plotlyjs=(i == 0)))
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
    source_hash = audit.json_hash([(c.cell_type, c.source) for c in book.cells])
    result_hash = digest(root / PUBLIC / "audit.json")
    if marker.exists():
        old = json.loads(marker.read_text())
        if (
            old["source_hash"] != source_hash
            or old["audit_sha256"] != result_hash
            or old["notebook_sha256"] != digest(path)
        ):
            raise ValueError("saved notebook identity changed; preserve it")
        return {**old, "status": "NOTEBOOK_REUSED", "cells_reexecuted": 0}
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
    plots = sum(
        "application/vnd.plotly.v1+json" in o.get("data", {}) for c in cells for o in c.outputs
    )
    if len(cells) != 8 or any(c.execution_count is None for c in cells) or plots != 8:
        raise ValueError("notebook did not complete its eight cells/charts")
    atomic_bytes(path, nbformat.writes(book).encode())
    record = {
        "status": "NOTEBOOK_EXECUTED",
        "code_cells": len(cells),
        "plotly_charts": plots,
        "source_hash": source_hash,
        "notebook_sha256": digest(path),
        "audit_sha256": result_hash,
    }
    atomic_json(marker, record)
    return record


def export(root):
    names = list(NEW_SOURCES) + [NOTEBOOK, PUBLIC + "/audit.json"]
    names += [
        PRIVATE + "/" + n
        for n in (
            "launcher_report.json",
            "tests.xml",
            "install.json",
            "notebook_execution.json",
            "replay.json",
            "last_invocation.json",
        )
    ]
    # Include public summaries only, never raw examples, vectors or private predictions.
    config = json.loads((root / CONFIG).read_text())
    names += [r["public"] + "/results.json" for r in config["rounds"]]
    data = {name: (root / name).read_bytes() for name in names if (root / name).is_file()}
    data["SHA256SUMS.json"] = json.dumps(
        {name: hashlib.sha256(value).hexdigest() for name, value in data.items()}, indent=2
    ).encode()
    target = Path.home() / "jigsaw_feature_value_return.zip"
    temporary = target.with_suffix(".partial")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as output:
        for name, value in data.items():
            output.writestr(name, value)
    os.replace(temporary, target)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--recover-reference", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--notebook-only", action="store_true")
    args = parser.parse_args()
    if args.export_only:
        print("RETURN_FILE:", export(args.root))
        return
    if not hasattr(signal, "setitimer"):
        raise ValueError("POSIX hard timer required")

    def expired(signum, frame):
        raise TimeoutError("audit runtime cap; completed files preserved")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 240)
    try:
        if args.notebook_only:
            first = execute_notebook(args.root)
            second = execute_notebook(args.root)
            atomic_json(
                args.root / PRIVATE / "replay.json",
                {
                    "first": first["status"],
                    "second": second["status"],
                    "cells_reexecuted": second["cells_reexecuted"],
                },
            )
        else:
            run_audit(args.root, recover=args.recover_reference)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    main()
