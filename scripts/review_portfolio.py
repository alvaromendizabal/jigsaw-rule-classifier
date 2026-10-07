"""Verify and re-execute the compact public review without models or cloud access."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = "notebooks/31_complete_project_review.ipynb"
FRONTIER_NOTEBOOK = "notebooks/29_five_model_frontier_review.ipynb"
INPUTS = (
    "reports/checkpoints/kaggle_submission.json",
    "reports/checkpoints/kaggle_adaptation.json",
    "reports/checkpoints/cross_model_frontier_20261006.json",
    "scripts/review_portfolio.py",
    "reports/checkpoints/five_model_frontier.json",
)
MIME = "application/vnd.plotly.v1+json"


def finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Evidence requires finite numeric values")
    return float(value)


def auc(value: object) -> float:
    number = finite(value)
    if not 0 <= number <= 1:
        raise ValueError("ROC AUC must be between zero and one")
    return number


def close(left: object, right: object, tolerance: float = 1e-9) -> None:
    if not math.isclose(finite(left), finite(right), rel_tol=0, abs_tol=tolerance):
        raise ValueError("Aggregate evidence arithmetic disagrees")


def evidence(root: Path = ROOT) -> dict:
    baseline, retained, frontier = [json.loads((root / p).read_text()) for p in INPUTS[:3]]
    original, accepted = baseline["submission"], retained["submission"]
    for item in (original, accepted):
        if item["status"] != "succeeded" or item["is_late"] is not True:
            raise ValueError("Review requires explicitly successful late-evaluation receipts")
    official = {
        "labels": [
            "Lexical · public",
            "Retained 4B · public",
            "Lexical · private",
            "Retained 4B · private",
        ],
        "values": [
            auc(original["public_score"]),
            auc(accepted["public_score"]),
            auc(original["private_score"]),
            auc(accepted["private_score"]),
        ],
    }
    for split in ("public", "private"):
        close(accepted[f"{split}_score"], frontier["official_retained"][f"{split}_roc_auc"])
        close(
            accepted[f"{split}_score"] - original[f"{split}_score"],
            retained["comparison"][f"{split}_auc_gain"],
        )
    cohort = frontier["evaluation"]
    if (
        cohort["untouched_holdout"] is not False
        or cohort["directly_comparable_to_official_score"] is not False
        or cohort["repeatedly_inspected_development_cohort"] is not True
    ):
        raise ValueError("Development scope must remain distinct from hidden evaluation")
    for field in ("rows", "policies"):
        if type(cohort[field]) is not int or cohort[field] <= 0:
            raise ValueError("Cohort dimensions must be positive integers")
    studies = {item["id"]: item for item in frontier["studies"]}
    if len(studies) != len(frontier["studies"]):
        raise ValueError("Duplicate study identifiers")
    native = auc(studies["E48"]["policy_macro_roc_auc"])
    reciprocal = studies["E31"]
    delta = auc(reciprocal["primary_auc"]) - auc(reciprocal["matched_native_control_auc"])
    close(delta, reciprocal["delta_vs_matched_control"])
    interval = list(map(finite, reciprocal["bootstrap_ci95"]))
    if len(interval) != 2 or not interval[0] <= delta <= interval[1]:
        raise ValueError("Reciprocal confidence interval is inconsistent")
    auc(reciprocal["bootstrap_positive_fraction"])
    if any(studies[key]["decision"] != "valid_negative" for key in ("E47", "E49", "E31")):
        raise ValueError("The published transfer studies were not promoted")
    return {
        "official": official,
        "private_gain": accepted["private_score"] - original["private_score"],
        "official_verified_utc": accepted["first_verified_scored_utc"],
        "frontier_receipt_utc": frontier["generated_utc"],
        "development_cohort": {"rows": cohort["rows"], "policies": cohort["policies"]},
        "development": {
            "labels": [
                "Fixed teacher − native",
                "Anchored transfer − native",
                "Reciprocal − peer control",
            ],
            "values": [
                auc(studies["E47"]["policy_macro_roc_auc"]) - native,
                auc(studies["E49"]["policy_macro_roc_auc"]) - native,
                delta,
            ],
        },
        "reciprocal_ci95": interval,
        "reciprocal_positive_fraction": reciprocal["bootstrap_positive_fraction"],
        "next_study": {
            "id": frontier["current_frontier"]["id"],
            "status": frontier["current_frontier"]["status"],
        },
        "scope": "Completed public evidence review; no training or new hidden evaluation",
    }


def frontier_values(root: Path = ROOT) -> dict:
    record = json.loads((root / INPUTS[-1]).read_text())
    full, global_result = record["full_oof"], record["global_deployment_validation"]
    values = [
        auc(full["qwen25_standalone_macro_auc"]),
        auc(global_result["prior_four_model_global_macro_auc"]),
        auc(full["fixed_insertion_macro_auc"]),
        auc(global_result["promoted_five_model_global_macro_auc"]),
    ]
    # This historical receipt independently rounds its candidate and gain to six decimals.
    close(values[-1] - values[1], global_result["gain"], tolerance=1e-6)
    return {
        "labels": [
            "Qwen2.5 standalone",
            "Prior 4-model global",
            "Fixed insertion",
            "Promoted 5-model global",
        ],
        "values": values,
    }


def source_digest(notebook) -> str:
    cells = [(cell.cell_type, cell.source) for cell in notebook.cells]
    return hashlib.sha256(json.dumps(cells, ensure_ascii=False).encode()).hexdigest()


def input_digests(root: Path = ROOT) -> dict:
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in INPUTS}


def verify_svg(svg: str, values: list, limits: tuple, origin: float) -> None:
    """Validate bar geometry as well as visible numbers in the static fallback."""
    root = ElementTree.fromstring(svg)
    rectangles = root.findall("{http://www.w3.org/2000/svg}rect")[1:]
    if len(rectangles) != len(values):
        raise ValueError("Static figure bar count disagrees with evidence")
    low, high = limits
    zero = origin + 620 * (0 - low) / (high - low)
    for rectangle, value in zip(rectangles, values, strict=True):
        edge = origin + 620 * (value - low) / (high - low)
        close(float(rectangle.attrib["width"]), abs(edge - zero), tolerance=0.006)
        close(float(rectangle.attrib["x"]), min(zero, edge), tolerance=0.006)


def verify(notebook, root: Path = ROOT) -> dict:
    expected = evidence(root)
    metadata = notebook.metadata.get("public_review", {})
    if metadata.get("source_sha256") != source_digest(notebook):
        raise ValueError("Notebook source changed after execution")
    if metadata.get("inputs_sha256") != input_digests(root):
        raise ValueError("Evidence inputs changed after execution")
    if metadata.get("engine") not in {"jupyter", "inprocess"}:
        raise ValueError("Execution engine is missing")
    cells = [c for c in notebook.cells if c.cell_type == "code" and c.source.strip()]
    if not cells or [c.execution_count for c in cells] != list(range(1, len(cells) + 1)):
        raise ValueError("Notebook contains incomplete or out-of-order execution")
    outputs = [o for c in cells for o in c.outputs]
    if any(
        o.output_type == "error"
        or (o.output_type == "stream" and o.name == "stderr" and o.text.strip())
        for o in outputs
    ):
        raise ValueError("Notebook execution emitted an error or stderr")
    text = "".join(o.get("text", "") for o in outputs)
    if metadata.get("kind") == "five_model":
        expected_frontier = frontier_values(root)
        figures = [o.data for o in outputs if MIME in o.get("data", {})]
        if len(figures) != 1 or len(figures[0][MIME].get("data", [])) != 1:
            raise ValueError("Expected one complete frontier figure")
        data = figures[0]
        trace = data[MIME]["data"][0]
        if (
            trace.get("x") != expected_frontier["values"]
            or trace.get("y") != expected_frontier["labels"]
        ):
            raise ValueError("Saved frontier figure disagrees with receipt")
        if data[MIME]["layout"]["xaxis"].get("range") != [0, 1.05]:
            raise ValueError("Frontier chart must use a stated zero-based scale")
        svg = data.get("image/svg+xml", "")
        if "<svg" not in svg or any(f"{v:.6f}" not in svg for v in expected_frontier["values"]):
            raise ValueError("Static frontier figure is absent or stale")
        verify_svg(svg, expected_frontier["values"], (0, 1.05), 250)
        if "FOLLOWING_CELL_EXECUTED" not in text:
            raise ValueError("Frontier notebook did not finish")
        return {
            "status": "verified",
            "engine": metadata["engine"],
            "code_cells": len(cells),
            "plotly": 1,
            "svg": 1,
            "scope": "Five-model development evidence only",
        }
    if metadata.get("kind") != "complete_review":
        raise ValueError("Unknown public review kind")
    marker = "PUBLIC_REVIEW_SUMMARY="
    summaries = [
        json.loads(line[len(marker) :]) for line in text.splitlines() if line.startswith(marker)
    ]
    if summaries != [expected]:
        raise ValueError("Saved numerical summary disagrees with current receipts")
    figures = [o.data for o in outputs if MIME in o.get("data", {})]
    if len(figures) != 2:
        raise ValueError("Expected two saved Plotly figures")
    for data, section in zip(figures, ("official", "development"), strict=True):
        figure = data[MIME]
        trace = figure["data"][0]
        if (
            trace.get("x") != expected[section]["values"]
            or trace.get("y") != expected[section]["labels"]
        ):
            raise ValueError("Saved figure disagrees with current receipts")
        svg = data.get("image/svg+xml", "")
        if "<svg" not in svg or any(f"{v:.6f}" not in svg for v in expected[section]["values"]):
            raise ValueError("Saved static figure is absent or stale")
        limits = (0, 1.05) if section == "official" else (-0.0007, 0.0007)
        verify_svg(svg, expected[section]["values"], limits, 300)
        if not 320 <= figure["layout"]["height"] <= 550:
            raise ValueError("Review figure dimensions are unsuitable")
    return {
        "status": "verified",
        "engine": metadata["engine"],
        "code_cells": len(cells),
        "plotly": len(figures),
        "svg": len(figures),
        "private_auc": expected["official"]["values"][-1],
        "scope": expected["scope"],
    }


def execute(root: Path, destination: Path, engine: str = "jupyter", frontier: bool = False) -> dict:
    notebook = nbformat.read(root / (FRONTIER_NOTEBOOK if frontier else NOTEBOOK), as_version=4)
    # Existing published outputs are never trusted as evidence of this execution.
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
    if engine == "jupyter":
        from jupyter_client import KernelManager
        from nbclient import NotebookClient

        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        NotebookClient(notebook, timeout=90, km=manager).execute(
            cwd=str(root), transport_encryption="required"
        )
    else:
        import os

        from scripts.execute_notebooks import execute_inprocess

        previous = Path.cwd()
        try:
            os.chdir(root)
            execute_inprocess(notebook, {})
        finally:
            os.chdir(previous)
    notebook.metadata["public_review"] = {
        "engine": engine,
        "kind": "five_model" if frontier else "complete_review",
        "completed_utc": datetime.now(UTC).isoformat(),
        "source_sha256": source_digest(notebook),
        "inputs_sha256": input_digests(root),
        "scope": "Public aggregate rendering only; no AWS or model execution",
    }
    verify(notebook, root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, destination)
    return verify(nbformat.read(destination, as_version=4), root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute", action="store_true", help="Run a real Jupyter kernel and save/reopen"
    )
    parser.add_argument(
        "--inprocess", action="store_true", help="Local diagnostic without Jupyter transport"
    )
    parser.add_argument(
        "--notebook", type=Path, help="Saved review notebook to verify or execution destination"
    )
    parser.add_argument("--frontier", action="store_true", help="Review the five-model notebook 29")
    args = parser.parse_args()
    selected = FRONTIER_NOTEBOOK if args.frontier else NOTEBOOK
    if args.execute and args.inprocess:
        parser.error("Choose actual Jupyter execution or the in-process diagnostic")
    if args.execute or args.inprocess:
        destination = args.notebook or ROOT / "runs/portfolio_review" / Path(selected).name
        result = execute(
            ROOT, destination, "inprocess" if args.inprocess else "jupyter", args.frontier
        )
        result["saved_notebook"] = (
            str(destination.relative_to(ROOT))
            if destination.is_relative_to(ROOT)
            else str(destination)
        )
    else:
        result = verify(nbformat.read(args.notebook or ROOT / selected, as_version=4))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    # Running a file under scripts/ does not automatically expose its sibling package.
    sys.path.insert(0, str(ROOT))
    main()
