"""Execute and checkpoint read-only portfolio notebooks; publish canonical files explicitly."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import nbformat
from filelock import FileLock
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

from jigsaw_rules.data import synthetic
from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest, stage


def source_hash(nb) -> str:
    """Ignore outputs and execution timestamps, never narrative or executable source."""
    source = {
        "kernel": nb.metadata.get("kernelspec"),
        "cells": [
            {
                **{k: c.get(k) for k in ("cell_type", "id", "source", "attachments")},
                "metadata": {k: v for k, v in c.metadata.items() if k != "execution"},
            }
            for c in nb.cells
        ],
    }
    return hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()


def select_notebooks(root: Path, requested: list[str] | None, mode: str) -> list[Path]:
    names = (
        "00_environment_and_data.ipynb",
        "01_data_and_validation.ipynb",
        "02_baseline_and_review.ipynb",
        "03_saved_results.ipynb",
        "04_semantic_benchmark.ipynb",
    )
    paths = [root / "notebooks" / name for name in names]
    if mode != "public":
        paths = [root / "kaggle/submission.ipynb"]
    selected = []
    for item in requested or [str(p.relative_to(root)) for p in paths]:
        matches = [p for p in paths if item in (str(p.relative_to(root)), p.name, p.name[:2])]
        if (
            len(matches) != 1
            or matches[0].is_symlink()
            or not matches[0].is_file()
            or not matches[0].resolve().is_relative_to(root.resolve())
        ):
            raise ValueError(f"Select an existing {mode} notebook by exact path or number: {item}")
        if matches[0] not in selected:
            selected.append(matches[0])
    if not selected:
        raise ValueError("No notebooks selected")
    return selected


def execution_contract(root: Path, work: Path, nb, mode: str, engine: str) -> dict:
    """Hash actual inputs so changed evidence cannot reuse stale display outputs."""
    paths = [*sorted((root / "src").rglob("*.py")), root / "scripts/execute_notebooks.py"]
    paths += [root / "pyproject.toml", root / "uv.lock"]
    if mode == "public":
        paths += [
            p
            for kind in ("baseline", "semantic")
            for p in sorted((root / "reports" / kind).rglob("*"))
            if p.suffix in (".json", ".svg")
        ]
    else:
        paths += sorted((work / "data/raw").glob("*.csv"))
        marker = work / "data/raw/SYNTHETIC.txt"
        if marker.exists():
            paths.append(marker)
    return {
        "schema": 1,
        "source": source_hash(nb),
        "mode": mode,
        "engine": engine,
        "python": sys.version.split()[0],
        "packages": {
            name: version(name)
            for name in (
                "numpy",
                "pandas",
                "scikit-learn",
                "scipy",
                "joblib",
                "plotly",
                "nbclient",
                "nbformat",
                "ipykernel",
            )
        },
        "inputs": {str(p.relative_to(root)): digest(p) for p in paths},
    }


def validate_execution(nb) -> None:
    cells = [c for c in nb.cells if c.cell_type == "code" and c.source.strip()]
    if not cells or any(c.execution_count is None for c in cells):
        raise ValueError("Notebook contains unexecuted code cells")
    for cell in cells:
        for output in cell.outputs:
            if output.output_type == "error":
                raise ValueError("Notebook contains an execution error")
            if output.output_type == "stream" and output.name == "stderr" and output.text.strip():
                raise ValueError("Notebook has stderr output; investigate before publication")


def publish_notebook(source: Path, executed: Path, expected_digest: str, mode: str) -> None:
    if mode != "public":
        raise ValueError("Only public aggregate notebooks may be published")
    nb = nbformat.read(executed, as_version=4)
    validate_execution(nb)
    if nb.metadata.get("verification", {}).get("mode") != "public":
        raise ValueError("Cached notebook is not public evidence")
    if digest(source) != expected_digest or source_hash(nb) != source_hash(
        nbformat.read(source, as_version=4)
    ):
        raise ValueError("Canonical notebook changed during execution; publication refused")
    atomic_bytes(source, executed.read_bytes())


def execute_inprocess(nb, env, on_start=None, on_done=None):
    """Test cell logic with rich outputs; this does not verify Jupyter transport."""
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output

    before, before_path = os.environ.copy(), sys.path[:]
    os.environ.update(env)
    sys.path.insert(0, str(Path.cwd()))
    InteractiveShell.clear_instance()
    shell = InteractiveShell.instance()
    count = 0
    try:
        for index, cell in enumerate(nb.cells):
            if cell.cell_type != "code" or not cell.source.strip():
                continue
            if on_start:
                on_start(cell=cell, cell_index=index)
            count += 1
            with capture_output() as captured:
                result = shell.run_cell(cell.source, store_history=True)
            if result.error_before_exec:
                raise result.error_before_exec
            if result.error_in_exec:
                raise result.error_in_exec
            cell.execution_count, cell.outputs = count, []
            for name, value in (("stdout", captured.stdout), ("stderr", captured.stderr)):
                if value:
                    cell.outputs.append(nbformat.v4.new_output("stream", name=name, text=value))
            for rich in captured.outputs:
                cell.outputs.append(
                    nbformat.v4.new_output("display_data", data=rich.data, metadata=rich.metadata)
                )
            if on_done:
                on_done(cell=cell, cell_index=index)
    finally:
        os.environ.clear()
        os.environ.update(before)
        sys.path[:] = before_path
        InteractiveShell.clear_instance()


def execute_one(root: Path, work: Path, path: Path, mode: str, engine: str, log: Progress) -> Path:
    nb = nbformat.read(path, as_version=4)
    contract = execution_contract(root, work, nb, mode, engine)
    key = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:20]
    directory = root / "runs/notebook_execution" / key

    def action(target: Path) -> None:
        env = os.environ.copy()
        env.update(
            {
                "JIGSAW_ROOT": str(root),
                "JIGSAW_CLOUD": "0",
                "JIGSAW_KAGGLE_INPUT": str(work / "data/raw"),
                "JIGSAW_KAGGLE_OUTPUT": str(target),
            }
        )
        with Progress(
            root / "logs/notebook_execution.jsonl", path.stem, total_started=log.started
        ) as progress:

            def started(cell, cell_index, **kwargs):
                if cell.cell_type == "code":
                    progress.emit("cell_started", cell=cell_index + 1, cells=len(nb.cells))

            def finished(cell, cell_index, **kwargs):
                progress.emit("cell_completed", cell=cell_index + 1, cells=len(nb.cells))

            if engine == "jupyter":
                kernel = directory / "kernel"
                atomic_json(
                    kernel / "kernel.json",
                    {
                        "argv": [
                            sys.executable,
                            "-m",
                            "ipykernel_launcher",
                            "-f",
                            "{connection_file}",
                        ],
                        "display_name": "Jigsaw verification",
                        "language": "python",
                        "metadata": {"supported_encryption": ["curve"]},
                    },
                )
                kernel_name = f"jigsaw-verification-{key}"
                manager = KernelSpecManager()
                manager.install_kernel_spec(str(kernel), kernel_name=kernel_name, user=True)
                try:
                    client = NotebookClient(
                        nb,
                        timeout=600,
                        kernel_name=kernel_name,
                        allow_errors=False,
                        on_cell_start=started,
                        on_cell_executed=finished,
                    )
                    client.execute(cwd=str(root), env=env, transport_encryption="required")
                finally:
                    manager.remove_kernel_spec(kernel_name)
            else:
                execute_inprocess(nb, env, started, finished)
            validate_execution(nb)
            nb.metadata["verification"] = {
                "mode": mode,
                "engine": engine,
                "synthetic": mode == "synthetic",
                "data_kind": "synthetic" if mode == "synthetic" else "competition",
                "source_sha256": source_hash(nb),
                "contract_sha256": key,
                "completed_at": datetime.now(UTC).isoformat(),
                "transport_encryption": "required" if engine == "jupyter" else "not_applicable",
                "scope": "aggregate evidence rendering"
                if mode == "public"
                else "offline inference",
            }
            atomic_bytes(target / path.name, nbformat.writes(nb).encode())
            atomic_json(target / "execution.json", contract)
            progress.emit("notebook_executed", notebook=path.name)

    committed = stage(directory, path.stem, action)
    cached = nbformat.read(committed / path.name, as_version=4)
    validate_execution(cached)
    log.emit("notebook_verified", notebook=path.name, mode=mode, checkpoint=key)
    return committed / path.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--synthetic", action="store_true", help="Test only offline Kaggle inference"
    )
    modes.add_argument(
        "--kaggle", action="store_true", help="Explicitly fit offline inference on real data"
    )
    parser.add_argument("--notebook", action="append", help="Exact path, filename, or 00–04")
    parser.add_argument(
        "--publish", action="store_true", help="Atomically update canonical notebooks"
    )
    parser.add_argument("--engine", choices=["jupyter", "inprocess"], default="jupyter")
    args = parser.parse_args()
    mode = "synthetic" if args.synthetic else "kaggle" if args.kaggle else "public"
    if args.publish and mode != "public":
        parser.error("--publish is only available for public aggregate notebooks")
    root = Path(__file__).resolve().parents[1]
    work = root / "runs/notebook_verification" if args.synthetic else root
    paths = select_notebooks(root, args.notebook, mode)
    if mode == "kaggle" and (work / "data/raw/SYNTHETIC.txt").exists():
        parser.error("--kaggle requires real competition data, not synthetic fixtures")
    (root / "runs").mkdir(exist_ok=True)
    with FileLock(str(root / "runs/notebook_execution.lock"), timeout=1):
        if args.synthetic and not (work / "data/raw/SYNTHETIC.txt").exists():
            synthetic(work / "data/raw")
        with Progress(root / "logs/notebook_execution.jsonl", "notebook_execution") as log:
            for path in paths:
                original = digest(path)
                executed = execute_one(root, work, path, mode, args.engine, log)
                if args.publish:
                    publish_notebook(path, executed, original, mode)
                    log.emit("notebook_published", notebook=path.name)
                if mode != "public":
                    for name in ("submission.csv", "submission_manifest.json"):
                        atomic_bytes(
                            work / "kaggle_output" / name, (executed.parent / name).read_bytes()
                        )
            log.emit("NOTEBOOKS_VERIFIED", notebooks=len(paths), mode=mode, published=args.publish)


if __name__ == "__main__":
    main()
