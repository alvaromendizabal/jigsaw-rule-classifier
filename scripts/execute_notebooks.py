"""Execute and checkpoint read-only portfolio notebooks; publish canonical files explicitly."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
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
    for name in (
        "build_research_report.py",
        "build_release_report.py",
        "build_expanded_report.py",
        "build_formatting_report.py",
        "build_model_report.py",
        "build_protected_report.py",
        "build_delivery_report.py",
    ):
        figure_builder = root / "scripts" / name
        if figure_builder.exists():
            paths.append(figure_builder)
    paths += [root / "pyproject.toml", root / "uv.lock"]
    paths += [p for p in sorted((root / "configs").glob("*.json")) if p.name != "local.json"]
    if mode == "public":
        coverage = root / "docs/FEATURE_COVERAGE.md"
        if coverage.exists():
            paths.append(coverage)
        paths += [
            p
            for kind in (
                "baseline",
                "semantic",
                "features",
                "research",
                "pairs",
                "sensitivity",
                "robustness",
                "instructions",
                "released",
                "expanded",
                "retrieval",
                "resolution",
                "formatting",
                "feature_decision",
                "model_validation",
                "confirmation",
                "delivery",
            )
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
                "JIGSAW_SUBMISSION_CACHE": str(root / "runs/submission_cache"),
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


PUBLIC_ORIGINS = {
    "https://github.com/alvaromendizabal/jigsaw-rule-classifier.git",
    "https://github.com/alvaromendizabal/jigsaw-rule-classifier",
    "git@github.com:alvaromendizabal/jigsaw-rule-classifier.git",
}


def push_publication(root: Path, branch: str) -> str:
    """Explicitly commit/push an allowlist; preserve private files and unrelated edits."""
    from jigsaw_rules.features import feature_evidence
    from scripts.build_notebooks import notebooks, same_sources

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()

    with Progress(root / "logs/notebook_execution.jsonl", "github_publication") as log:
        if not branch.startswith("results/"):
            raise ValueError("Publication requires a dedicated results/... branch")
        git("check-ref-format", "--branch", branch)
        if git("remote", "get-url", "origin") not in PUBLIC_ORIGINS:
            raise ValueError("Origin is not the expected Jigsaw repository")
        if git("diff", "--cached", "--name-only"):
            raise ValueError("Preserve existing staged changes before publication")
        current = git("branch", "--show-current")
        if current not in {"main", branch}:
            raise ValueError("Start publication on main or the requested results branch")
        paths = [str(p.relative_to(root)) for p in select_notebooks(root, None, "public")]
        expected = notebooks()
        payloads = {}
        for name in paths:
            payloads[name] = (root / name).read_bytes()
            nb = nbformat.reads(payloads[name].decode(), as_version=4)
            validate_execution(nb)
            if not same_sources(nb, expected[name]):
                raise ValueError("Rebuild canonical notebook sources before publication")
            verification = nb.metadata.get("verification", {})
            if verification.get("mode") != "public" or verification.get("engine") != "jupyter":
                raise ValueError("Publication requires actual public Jupyter execution")
            contract = execution_contract(root, root, nb, "public", "jupyter")
            key = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:20]
            if verification.get("contract_sha256") != key:
                raise ValueError("Notebook evidence is stale; execute all five before pushing")
        evidence = feature_evidence(root)
        if evidence is not None:
            source = {p.name: digest(p) for p in (root / "src/jigsaw_rules").glob("*.py")}
            if source != evidence["metadata"]["source_sha256"]:
                # Historical evidence can be re-rendered only if it is already committed
                # byte-for-byte. It cannot be relabeled as an experiment on current code.
                for name in ("metadata.json", *evidence["metadata"]["files"]):
                    report = f"reports/features/{name}"
                    previous = subprocess.run(
                        ["git", "show", f"HEAD:{report}"], cwd=root, capture_output=True
                    )
                    if previous.returncode or previous.stdout != (root / report).read_bytes():
                        raise ValueError("Uncommitted historical feature evidence needs review")
            paths += ["reports/features/metadata.json"]
            paths += ["reports/features/" + name for name in evidence["metadata"]["files"]]
            for name in paths:
                if name not in payloads:
                    payloads[name] = (root / name).read_bytes()
            for name, sha in evidence["metadata"]["files"].items():
                if hashlib.sha256(payloads["reports/features/" + name]).hexdigest() != sha:
                    raise ValueError("Feature files changed during publication")
        from jigsaw_rules.diagnostics import diagnostic_evidence
        from jigsaw_rules.expanded import expanded_evidence
        from jigsaw_rules.feature_decision import decision_evidence
        from jigsaw_rules.formatting import formatting_evidence
        from jigsaw_rules.instructions import instruction_evidence
        from jigsaw_rules.pairs import pairs_evidence
        from jigsaw_rules.released import released_evidence
        from jigsaw_rules.research import research_evidence
        from jigsaw_rules.resolution import resolution_evidence
        from jigsaw_rules.retrieval import retrieval_evidence
        from jigsaw_rules.robustness import robustness_evidence

        readers = {
            "research": research_evidence,
            "pairs": pairs_evidence,
            "sensitivity": diagnostic_evidence,
            "robustness": robustness_evidence,
            "instructions": instruction_evidence,
            "released": released_evidence,
            "expanded": expanded_evidence,
            "retrieval": retrieval_evidence,
            "resolution": resolution_evidence,
            "formatting": formatting_evidence,
            "feature_decision": decision_evidence,
        }
        for kind, reader in readers.items():
            current_evidence = reader(root)
            if current_evidence is None:
                continue
            for name in ("metadata.json", *current_evidence["metadata"]["files"]):
                report = f"reports/{kind}/{name}"
                paths.append(report)
                payloads[report] = (root / report).read_bytes()
        if (root / "reports/formatting/figures.json").exists():
            from scripts.build_formatting_report import verify_figures as verify_formatting_figures

            manifest = verify_formatting_figures(root)
            for name in ("figures.json", *manifest["files"]):
                report = f"reports/formatting/{name}"
                paths.append(report)
                payloads[report] = (root / report).read_bytes()
        if (root / "reports/expanded/figures.json").exists():
            from scripts.build_expanded_report import verify_figures as verify_expanded_figures

            manifest = verify_expanded_figures(root)
            for name in ("figures.json", *manifest["files"]):
                report = f"reports/expanded/{name}"
                paths.append(report)
                payloads[report] = (root / report).read_bytes()
        if (root / "reports/released/figures.json").exists():
            from scripts.build_release_report import verify_figures

            manifest = verify_figures(root)
            for name in ("figures.json", *manifest["files"]):
                report = f"reports/released/{name}"
                paths.append(report)
                payloads[report] = (root / report).read_bytes()
        manifest_path = root / "reports/research/figures.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["source_sha256"] != digest(root / "scripts/build_research_report.py"):
                raise ValueError("Research figures use outdated rendering source")
            allowed_figures = {
                f"{name}.{suffix}"
                for name in ("ablation", "screening", "stability")
                for suffix in ("svg", "plotly.json")
            }
            if set(manifest["files"]) != allowed_figures:
                raise ValueError("Research figure allowlist differs")
            for name, sha in manifest["files"].items():
                report = f"reports/research/{name}"
                if digest(root / report) != sha:
                    raise ValueError("Research figure checksum differs")
                paths.append(report)
                payloads[report] = (root / report).read_bytes()
            paths.append("reports/research/figures.json")
            payloads["reports/research/figures.json"] = manifest_path.read_bytes()
        outside = set(git("diff", "--name-only").splitlines()) - set(paths)
        untracked_source = git("ls-files", "--others", "--exclude-standard", "--", "src", "scripts")
        if outside or untracked_source:
            raise ValueError("Unpublished source or unrelated edits exist; they were not changed")
        if current != branch:
            existing = git("branch", "--list", branch)
            if existing:
                if git("rev-parse", branch) != git("rev-parse", "HEAD"):
                    raise ValueError("Results branch exists at another commit; choose a new name")
                git("switch", branch)
            else:
                git("switch", "-c", branch)
        git("add", "--", *paths)
        git("diff", "--cached", "--check")
        staged = set(git("diff", "--cached", "--name-only").splitlines())
        if not staged.issubset(paths):
            raise ValueError("Staged files exceed the public allowlist; push refused")
        for name in staged:
            indexed = subprocess.run(
                ["git", "show", ":" + name], cwd=root, check=True, capture_output=True
            ).stdout
            if indexed != payloads[name]:
                raise ValueError("Public file changed while staging; push refused")
        if staged:
            run_id = evidence["metadata"]["run_id"] if evidence else "recorded-reference-review"
            git(
                "commit",
                "-m",
                "results: publish verified notebook evidence",
                "-m",
                f"Evidence run: {run_id}. Canonical notebooks executed in Jupyter; "
                "only checksummed aggregate reports are included. Private data, predictions, "
                "credentials, submissions, and checkpoints remain excluded. "
                "No automatic model promotion or Kaggle upload.",
            )
            log.emit("public_commit_created", files=len(staged))
        commit = git("rev-parse", "HEAD")
        if staged and not set(
            git("diff-tree", "--no-commit-id", "--name-only", "-r", commit).splitlines()
        ).issubset(paths):
            raise ValueError("Concurrent staged changes entered the commit; push refused")
        git("push", "--set-upstream", "origin", branch)
        log.emit("GITHUB_PUSHED", branch=branch, commit=commit)
        print("Open a results pull request on GitHub; merge only after Quality passes.")
        return commit


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
    parser.add_argument(
        "--push-branch",
        help="Explicitly commit/push verified public outputs to a results/... branch",
    )
    args = parser.parse_args()
    if args.push_branch and (not args.publish or args.kaggle or args.synthetic):
        parser.error("--push-branch requires --publish in public mode")
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

    if args.push_branch:
        push_publication(root, args.push_branch)


if __name__ == "__main__":
    main()
