"""Execute every notebook; synthetic mode is an isolated, visibly labeled software test."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nbformat
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

from jigsaw_rules.data import synthetic
from jigsaw_rules.runtime import Progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--engine", choices=["jupyter", "inprocess"], default="jupyter")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    work = root / "runs/notebook_verification" if args.synthetic else root
    work.mkdir(parents=True, exist_ok=True)
    if args.synthetic and not (work / "data/raw/SYNTHETIC.txt").exists():
        synthetic(work / "data/raw")
    env = os.environ.copy()
    env.update(
        {
            "JIGSAW_ROOT": str(work),
            "JIGSAW_CLOUD": "0" if args.synthetic else "1",
            "JIGSAW_KAGGLE_INPUT": str(work / "data/raw"),
            "JIGSAW_KAGGLE_OUTPUT": str(work / "kaggle_output"),
        }
    )
    # A private kernelspec ensures execution uses this exact locked interpreter.
    import json

    kernel_dir = work / "kernel"
    kernel_dir.mkdir(exist_ok=True)
    (kernel_dir / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Jigsaw verification",
                "language": "python",
                "metadata": {"supported_encryption": ["curve"]},
            }
        )
    )
    if args.engine == "jupyter":
        manager = KernelSpecManager()
        manager.install_kernel_spec(str(kernel_dir), kernel_name="jigsaw-verification", user=True)
    output = work / "executed_notebooks"
    output.mkdir(exist_ok=True)
    paths = [*sorted((root / "notebooks").glob("*.ipynb")), root / "kaggle/submission.ipynb"]
    with Progress(root / "logs/notebook_execution.jsonl", "notebook_execution") as log:
        for path in paths:
            log.emit(
                "notebook_started", notebook=path.name, synthetic=args.synthetic, engine=args.engine
            )
            nb = nbformat.read(path, as_version=4)
            if args.engine == "jupyter":
                client = NotebookClient(
                    nb, timeout=600, kernel_name="jigsaw-verification", allow_errors=False
                )
                client.execute(cwd=str(root), env=env, transport_encryption="required")
            else:
                execute_inprocess(nb, env)
            nb.metadata["verification"] = {
                "engine": args.engine,
                "synthetic": args.synthetic,
                "transport_encryption": "required"
                if args.engine == "jupyter"
                else "not_applicable",
            }
            nbformat.write(nb, output / path.name)
            log.emit("notebook_completed", notebook=path.name)
    print(f"EXECUTED_NOTEBOOKS {output}")


def execute_inprocess(nb, env):
    """Execute real cells with captured rich outputs; does not test kernel transport."""
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output

    before = os.environ.copy()
    os.environ.update(env)
    InteractiveShell.clear_instance()
    shell = InteractiveShell.instance()
    count = 0
    try:
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            count += 1
            with capture_output() as captured:
                result = shell.run_cell(cell.source, store_history=True)
            if result.error_before_exec:
                raise result.error_before_exec
            if result.error_in_exec:
                raise result.error_in_exec
            cell.execution_count = count
            cell.outputs = []
            for name, value in [("stdout", captured.stdout), ("stderr", captured.stderr)]:
                if value:
                    cell.outputs.append(nbformat.v4.new_output("stream", name=name, text=value))
            for rich in captured.outputs:
                cell.outputs.append(
                    nbformat.v4.new_output("display_data", data=rich.data, metadata=rich.metadata)
                )
    finally:
        os.environ.clear()
        os.environ.update(before)
        InteractiveShell.clear_instance()


if __name__ == "__main__":
    main()
