"""Read-only original-data profiling and one checkpointed Plotly notebook.

This module does not fit a model, invoke AWS, download data, or call an encoder.
Public outputs contain aggregate counts, not comments or row-level labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from filelock import FileLock
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient
from nbformat.sign import NotebookNotary
from plotly.offline import get_plotlyjs

from jigsaw_rules.data import EXAMPLES, load_data, normalize, validate_frame
from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest
from jigsaw_rules.splits import purged_split

BASE_COMMIT = "824e92bfae7414aa156f0bf3195ba677a1f55b21"
KERNEL = "jigsaw-rules"
NOTEBOOK = "notebooks/05_data_readiness.ipynb"
RAW_HASHES = {
    "train.csv": "83948d06a1e4b16421b738add60ef489cf1d44a2349ca711958fbb41c6207a0a",
    "test.csv": "107b8b329200171f55d2523e6d32670641b53d5be41e0c1d5b9da0b8813cdd41",
    "sample_submission.csv": "6524e86a03abe387f0867247629e08493770a6f9d37120cc8165322e7486a066",
}
PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "plotly",
    "nbformat",
    "nbclient",
    "ipykernel",
    "jupyter-client",
)
CHARTS = (
    "class_balance",
    "length_distribution",
    "duplicate_burden",
    "query_eligibility",
    "training_purge",
    "support_conflicts",
)


def timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def repository_root(start: Path | None = None) -> Path:
    path = Path(start or os.environ.get("JIGSAW_ROOT", Path.cwd())).resolve()
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src/jigsaw_rules/data.py"
        ).is_file():
            return candidate
    raise ValueError("Open this notebook inside the Jigsaw repository")


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", *args],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    ).stdout.strip()


def raw_identity(root: Path) -> dict:
    observed = {}
    for name, expected in RAW_HASHES.items():
        path = root / "data/raw" / name
        if path.is_symlink() or not path.is_file() or digest(path) != expected:
            raise ValueError("Original-data hash mismatch or missing file: " + name)
        observed[name] = {"sha256": expected, "bytes": path.stat().st_size}
    return observed


def notebook_source(nb) -> str:
    value = [{"cell_type": c.cell_type, "source": c.source, "id": c.id} for c in nb.cells]
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def runtime_identity() -> dict:
    return {"python": platform.python_version(), "packages": {p: version(p) for p in PACKAGES}}


def readiness_contract(root: Path) -> dict:
    paths = [
        "pyproject.toml",
        "uv.lock",
        "src/jigsaw_rules/data.py",
        "src/jigsaw_rules/splits.py",
        "src/jigsaw_rules/runtime.py",
        "scripts/notebook_readiness.py",
    ]
    if (
        git(root, "rev-parse", "HEAD") != BASE_COMMIT
        or git(root, "branch", "--show-current") != "main"
    ):
        raise ValueError("Checkout differs from the verified main; preserve changes and reconcile")
    return {
        "schema": 1,
        "base_commit": BASE_COMMIT,
        "raw": raw_identity(root),
        "runtime": runtime_identity(),
        "source": {p: digest(root / p) for p in paths},
        "notebook_source": notebook_source(nbformat.read(root / NOTEBOOK, as_version=4)),
        "scope": "original_training_descriptive_profile_no_model_fits",
    }


def policy_names(rules) -> dict:
    result = {}
    for index, rule in enumerate(sorted(set(rules))):
        key = normalize(rule)
        name = (
            "Advertising"
            if "advertis" in key
            else "Legal advice"
            if "legal advice" in key
            else f"Policy {index + 1}"
        )
        if name in result.values():
            name = f"{name} ({index + 1})"
        result[rule] = name
    return result


def profile_frames(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame) -> dict:
    "Aggregate only. This uses original TRAIN labels descriptively, never test labels."
    validate_frame(train, train=True)
    validate_frame(test, train=False)
    if list(sample.columns) != ["row_id", "rule_violation"] or list(test.row_id) != list(
        sample.row_id
    ):
        raise ValueError("Sample/test schema or row order differs")
    if set(train.row_id) & set(test.row_id):
        raise ValueError("Training and preview IDs overlap")
    normalized = train.copy()
    normalized["rule"] = normalized.rule.map(normalize)
    names = policy_names(normalized.rule)
    policies, lengths, folds, conflicts = [], [], [], []
    bins = np.array([0, 80, 160, 320, 640, 1280, 2560, 5120, np.inf])
    labels = [
        "0–79",
        "80–159",
        "160–319",
        "320–639",
        "640–1,279",
        "1,280–2,559",
        "2,560–5,119",
        "5,120+",
    ]
    for rule in sorted(names):
        subset = normalized.loc[normalized.rule == rule]
        body = subset.body.map(normalize)
        label_counts = subset.rule_violation.value_counts()
        policies.append(
            {
                "policy": names[rule],
                "rows": int(len(subset)),
                "permitted": int(label_counts.get(0, 0)),
                "violating": int(label_counts.get(1, 0)),
                "violation_rate": float(subset.rule_violation.mean()),
                "communities": int(subset.subreddit.nunique()),
                "unique_normalized_bodies": int(body.nunique()),
                "repeated_body_occurrences": int(body.duplicated().sum()),
            }
        )
        counts, _ = np.histogram(subset.body.str.len().to_numpy(), bins=bins)
        lengths.extend(
            {"policy": names[rule], "characters": label, "rows": int(count)}
            for label, count in zip(labels, counts, strict=True)
        )
        pieces = [
            pd.DataFrame(
                {"text": subset.body.map(normalize), "label": subset.rule_violation.to_numpy()}
            )
        ]
        for column in EXAMPLES:
            pieces.append(
                pd.DataFrame(
                    {
                        "text": subset[column].map(normalize),
                        "label": int(column.startswith("positive")),
                    }
                )
            )
        expanded = pd.concat(pieces, ignore_index=True)
        counts_by_text = expanded.groupby("text").label.agg(["nunique", "size"])
        bad = counts_by_text["nunique"] > 1
        conflicts.append(
            {
                "policy": names[rule],
                "unique_labeled_texts": int(len(counts_by_text)),
                "conflicting_text_pairs": int(bad.sum()),
                "conflicting_occurrences": int(counts_by_text.loc[bad, "size"].sum()),
            }
        )

        tr = np.flatnonzero(normalized.rule.to_numpy() != rule)
        va = np.flatnonzero(normalized.rule.to_numpy() == rule)
        known = {normalize(text) for column in EXAMPLES for text in subset[column]}
        novel = ~subset.body.map(normalize).isin(known).to_numpy()
        queries = va[novel]
        record = {
            "policy": names[rule],
            "candidate_queries": int(len(va)),
            "support_known_queries": int((~novel).sum()),
            "novel_queries": int(len(queries)),
            "training_before": int(len(tr)),
        }
        try:
            if not len(queries):
                raise ValueError("No novel queries remain")
            kept, _, removed = purged_split(normalized, tr, queries)
            record.update(
                training_retained=int(len(kept)),
                training_purged=int(removed),
                status="eligible_for_later_protocol_review",
            )
        except ValueError as exc:
            record.update(
                training_retained=None, training_purged=None, status="blocked", reason=str(exc)
            )
        folds.append(record)
    missing = []
    for title, frame in (("train", train), ("preview_test", test), ("sample", sample)):
        for column in frame:
            missing.append(
                {"file": title, "column": column, "missing": int(frame[column].isna().sum())}
            )
    return {
        "schema": 1,
        "scope": "original_training_descriptive_profile",
        "file_rows": {
            "train.csv": int(len(train)),
            "test.csv": int(len(test)),
            "sample_submission.csv": int(len(sample)),
        },
        "policies": policies,
        "length_bins": lengths,
        "fold_preflight": folds,
        "support_conflicts": conflicts,
        "missingness": missing,
        "train_preview_body_overlap": int(
            len(set(train.body.map(normalize)) & set(test.body.map(normalize)))
        ),
        "training_labels_used_for_descriptive_counts": True,
        "test_target_labels_used": False,
        "model_fits": 0,
        "new_model_metric": None,
        "model_forward_passes": 0,
        "feature_selection_performed": False,
        "limitations": [
            "Counts are a data audit, not feature-value evidence or a model score.",
            (
                "Known-support query exclusion and exact-text purging do not "
                "prove semantic isolation."
            ),
            ("Original-data development has been inspected before; it is not a fresh holdout."),
            (
                "Conflict counts use training bodies plus their four supplied "
                "supports, not every historical study's population."
            ),
            "No raw comment text, community names, IDs, or row-level labels are exported.",
        ],
    }


def build_profile(root: Path) -> dict:
    raw = raw_identity(root)
    train, test, sample = load_data(root / "data/raw")
    summary = profile_frames(train, test, sample)
    summary["raw_identity"] = raw
    return summary


def figure(summary: dict, name: str) -> go.Figure:
    if name not in CHARTS:
        raise ValueError("Unknown chart: " + name)
    fig = go.Figure()
    if name == "class_balance":
        rows = summary["policies"]
        for key, label in (("permitted", "Permitted"), ("violating", "Violating")):
            fig.add_bar(name=label, x=[r["policy"] for r in rows], y=[r[key] for r in rows])
        title, axis = "Original training labels by policy", "Comments"
    elif name == "length_distribution":
        rows = summary["length_bins"]
        for policy in dict.fromkeys(r["policy"] for r in rows):
            subset = [r for r in rows if r["policy"] == policy]
            fig.add_bar(
                name=policy, x=[r["characters"] for r in subset], y=[r["rows"] for r in subset]
            )
        title, axis = "Comment length: character counts, not token counts", "Comments"
        fig.update_xaxes(title="Characters (fixed bins)")
    elif name == "duplicate_burden":
        rows = summary["policies"]
        for key, label in (
            ("unique_normalized_bodies", "Unique normalized texts"),
            ("repeated_body_occurrences", "Repeated occurrences"),
        ):
            fig.add_bar(name=label, x=[r["policy"] for r in rows], y=[r[key] for r in rows])
        title, axis = "Exact-text repetition within each policy", "Training rows"
    elif name == "query_eligibility":
        rows = summary["fold_preflight"]
        for key, label in (
            ("novel_queries", "Not in supplied support pool"),
            ("support_known_queries", "Already supplied as support"),
        ):
            fig.add_bar(name=label, x=[r["policy"] for r in rows], y=[r[key] for r in rows])
        title, axis = "Proposed query eligibility before fitting", "Held-out-policy comments"
    elif name == "training_purge":
        rows = summary["fold_preflight"]
        for key, label in (
            ("training_retained", "Retained"),
            ("training_purged", "Removed: query text in body/support"),
        ):
            fig.add_bar(name=label, x=[r["policy"] for r in rows], y=[r[key] for r in rows])
        title, axis = (
            "Training-context purge for each proposed holdout",
            "Other-policy training rows",
        )
    else:
        rows = summary["support_conflicts"]
        fig.add_bar(
            name="Conflicting labeled texts",
            x=[r["policy"] for r in rows],
            y=[r["conflicting_text_pairs"] for r in rows],
        )
        title, axis = (
            "Label conflicts in training bodies and supplied supports",
            "Unique normalized text/policy pairs",
        )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=460,
        barmode="group" if name == "length_distribution" else "stack",
        margin=dict(l=65, r=25, t=95, b=85),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        font=dict(family="Arial", size=13),
        yaxis_title=axis,
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


def save_profile(root: Path, summary: dict, figures: list[go.Figure]) -> dict:
    if len(figures) != len(CHARTS):
        raise ValueError("All six charts are required")
    output = root / "reports/data_readiness"
    summary_path, html_path = output / "summary.json", output / "dashboard.html"
    atomic_json(summary_path, summary)
    parts = [
        ("<!doctype html><html><head><meta charset='utf-8'><title>Jigsaw · data readiness</title>"),
        (
            "<style>body{font-family:Arial,sans-serif;max-width:1100px;margin"
            ":36px auto;padding:0 22px;}p{line-height:1.6}</style>"
        ),
        "<script>" + get_plotlyjs() + "</script></head><body>",
        (
            "<h1>Jigsaw · original-data readiness</h1><p>Descriptive checks "
            "only. No model fits, feature selections or new leaderboard "
            "score. The downloadable test file is a preview, not the hidden "
            "evaluation set.</p>"
        ),
    ]
    for i, plot in enumerate(figures):
        parts.append(
            pio.to_html(
                plot,
                full_html=False,
                include_plotlyjs=False,
                div_id=f"readiness-{i}",
                config={"responsive": True, "displaylogo": False},
            )
        )
    parts.append(
        "<p>Exact-text purging is not semantic-paraphrase isolation. "
        "These data have been inspected before. Feature research remains "
        "open.</p></body></html>"
    )
    atomic_bytes(html_path, "\n".join(parts).encode())
    receipt = {
        "status": "DATA_PROFILE_SAVED",
        "plots": len(figures),
        "model_fits": 0,
        "new_model_metric": None,
        "files": {str(p.relative_to(root)): digest(p) for p in (summary_path, html_path)},
    }
    atomic_json(root / "runs/notebook_readiness/profile_receipt.json", receipt)
    return receipt


def validate_executed(nb) -> int:
    count, charts = 0, 0
    for cell in nb.cells:
        if cell.cell_type != "code" or not cell.source.strip():
            continue
        count += 1
        if cell.execution_count is None:
            raise ValueError("A code cell is unexecuted")
        for output in cell.get("outputs", []):
            if output.output_type == "error":
                raise ValueError("Notebook error output")
            if output.output_type == "stream" and output.name == "stderr" and output.text.strip():
                raise ValueError("Notebook stderr output requires inspection")
            if "application/vnd.plotly.v1+json" in output.get("data", {}):
                charts += 1
    if charts != 6 or count < 7:
        raise ValueError("Notebook must contain six executed Plotly outputs")
    return count


def verify_checkpoint(root: Path, contract: dict) -> dict | None:
    marker = root / "runs/notebook_readiness/complete.json"
    if not marker.exists():
        return None
    saved = json.loads(marker.read_text())
    if saved["contract"] != contract:
        raise ValueError("Saved notebook contract differs; preserve it and inspect")
    for name, sha in saved["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != sha:
            raise ValueError("Completed notebook artifact is corrupt or changed")
    validate_executed(nbformat.read(root / NOTEBOOK, as_version=4))
    return saved


def execute_notebook(root: Path) -> dict:
    root = root.resolve()
    contract = readiness_contract(root)
    directory = root / "runs/notebook_readiness"
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / "execution.lock"), timeout=1):
        with Progress(directory / "events.jsonl", "readiness_notebook") as log:
            saved = verify_checkpoint(root, contract)
            if saved is not None:
                log.emit("verified_replay", cells_reexecuted=0, plots=6)
                return {"status": "NOTEBOOK_REUSED", "cells_reexecuted": 0, **saved["result"]}
            spec = KernelSpecManager().get_kernel_spec(KERNEL)
            if Path(spec.argv[0]).absolute() != Path(sys.executable).absolute():
                raise ValueError("Registered kernel points outside this project environment")
            path = root / NOTEBOOK
            old_hash = digest(path)
            nb = nbformat.read(path, as_version=4)

            def started(cell, cell_index, **kwargs):
                if cell.cell_type == "code":
                    log.emit("cell_started", cell=cell_index + 1, cells=len(nb.cells))

            def finished(cell, cell_index, **kwargs):
                if cell.cell_type == "code":
                    log.emit("cell_completed", cell=cell_index + 1, cells=len(nb.cells))

            client = NotebookClient(
                nb,
                kernel_name=KERNEL,
                timeout=45,
                startup_timeout=45,
                allow_errors=False,
                on_cell_start=started,
                on_cell_executed=finished,
            )
            env = {
                **os.environ,
                "JIGSAW_ROOT": str(root),
                "JIGSAW_CLOUD": "0",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            }
            # Preserve the repository's encrypted Jupyter requirement. No fallback.
            client.execute(cwd=str(root), env=env, transport_encryption="required")
            cells = validate_executed(nb)
            if digest(path) != old_hash or notebook_source(nb) != contract["notebook_source"]:
                raise ValueError("Notebook changed during execution; refusing overwrite")
            nb.metadata["jigsaw_readiness"] = {
                "executed_utc": timestamp(),
                "engine": "jupyter",
                "transport_encryption": "required",
                "model_fits": 0,
            }
            atomic_bytes(path, nbformat.writes(nb).encode())
            NotebookNotary().sign(nb)
            outputs = [
                NOTEBOOK,
                "reports/data_readiness/summary.json",
                "reports/data_readiness/dashboard.html",
                "runs/notebook_readiness/profile_receipt.json",
            ]
            result = {
                "code_cells": cells,
                "plotly_outputs": 6,
                "model_fits": 0,
                "new_model_metric": None,
                "notebook": NOTEBOOK,
            }
            saved = {
                "contract": contract,
                "files": {p: digest(root / p) for p in outputs},
                "result": result,
            }
            atomic_json(directory / "complete.json", saved)
            verify_checkpoint(root, contract)
            log.emit("notebook_verified", **result)
            return {"status": "NOTEBOOK_EXECUTED", **result}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    def stop(signum, frame):
        raise TimeoutError("Notebook execution interrupted; completed outputs preserved")

    signal.signal(signal.SIGTERM, stop)
    first = execute_notebook(args.root)
    second = execute_notebook(args.root)
    if second["status"] != "NOTEBOOK_REUSED" or second["cells_reexecuted"] != 0:
        raise ValueError("Replay did not reuse completed notebook")
    receipt = {"first_pass": first, "replay": second, "completed_utc": timestamp()}
    atomic_json(args.root / "runs/notebook_readiness/execution_receipt.json", receipt)
    print("RESULT: NOTEBOOK_AND_PLOTLY_PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
