"""Use local bare Git repositories; these tests never contact GitHub."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import nbformat
import pytest

from scripts.execute_notebooks import execution_contract, push_publication

ROOT = Path(__file__).resolve().parents[1]


def git(root, *arguments):
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path, monkeypatch):
    root, remote = tmp_path / "workspace", tmp_path / "origin.git"
    root.mkdir()
    for directory in ("src", "scripts", "reports", "notebooks", "configs"):
        shutil.copytree(ROOT / directory, root / directory)
    for filename in ("pyproject.toml", "uv.lock", ".gitignore"):
        shutil.copyfile(ROOT / filename, root / filename)
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "Authored test")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "add", ".")
    git(root, "commit", "-m", "Synthetic test repository")
    git(tmp_path, "init", "--bare", str(remote))
    git(root, "remote", "add", "origin", str(remote))
    git(root, "push", "-u", "origin", "main")
    monkeypatch.setattr("scripts.execute_notebooks.PUBLIC_ORIGINS", {str(remote)})
    for path in (root / "notebooks").glob("*.ipynb"):
        nb = nbformat.read(path, as_version=4)
        # Authored publication fixture, not evidence of a model/kernel execution.
        for index, cell in enumerate(nb.cells):
            if cell.cell_type == "code":
                cell.execution_count, cell.outputs = index + 1, []
        contract = execution_contract(root, root, nb, "public", "jupyter")
        nb.metadata["verification"] = {
            "mode": "public",
            "engine": "jupyter",
            "contract_sha256": hashlib.sha256(
                json.dumps(contract, sort_keys=True).encode()
            ).hexdigest()[:20],
        }
        nbformat.write(nb, path)
    return root, remote


def test_scoped_commit_and_repeated_push_are_idempotent(repository):
    root, remote = repository
    (root / "data/raw").mkdir(parents=True)
    (root / "data/raw/private.csv").write_text("private fixture")
    first = push_publication(root, "results/test")
    assert push_publication(root, "results/test") == first
    assert git(remote, "rev-parse", "results/test") == first
    assert "data/raw/private.csv" not in git(remote, "ls-tree", "-r", "--name-only", "results/test")
    assert (root / "data/raw/private.csv").read_text() == "private fixture"


@pytest.mark.parametrize("branch", ["main", "feat/arbitrary", "results/../../main"])
def test_unsafe_publication_branch_is_rejected(repository, branch):
    root, _ = repository
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        push_publication(root, branch)
    assert git(root, "branch", "--show-current") == "main"


def test_existing_staged_changes_are_preserved(repository):
    root, _ = repository
    path = root / "my-notes.txt"
    path.write_text("private authored note")
    git(root, "add", "my-notes.txt")
    with pytest.raises(ValueError, match="staged"):
        push_publication(root, "results/test")
    assert git(root, "diff", "--cached", "--name-only") == "my-notes.txt"


def test_changed_notebook_source_cannot_be_pushed(repository):
    root, _ = repository
    path = root / "notebooks/03_saved_results.ipynb"
    nb = nbformat.read(path, as_version=4)
    nb.cells[0].source = "Unreviewed replacement"
    nbformat.write(nb, path)
    with pytest.raises(ValueError, match="canonical"):
        push_publication(root, "results/test")
    assert not git(root, "branch", "--list", "results/test")


def test_changed_aggregate_evidence_requires_new_execution(repository):
    root, _ = repository
    path = root / "reports/baseline/results.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="stale"):
        push_publication(root, "results/test")


def test_wrong_origin_is_rejected_before_any_commit(repository, monkeypatch):
    root, _ = repository
    monkeypatch.setattr("scripts.execute_notebooks.PUBLIC_ORIGINS", set())
    before = git(root, "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="Origin"):
        push_publication(root, "results/test")
    assert git(root, "rev-parse", "HEAD") == before


def test_failed_push_preserves_commit_for_retry(repository, monkeypatch):
    root, remote = repository
    real_run = subprocess.run

    def fail_push(command, *args, **kwargs):
        if command[:2] == ["git", "push"]:
            raise subprocess.CalledProcessError(1, command, stderr="authored offline test")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr("scripts.execute_notebooks.subprocess.run", fail_push)
    with pytest.raises(subprocess.CalledProcessError):
        push_publication(root, "results/retry")
    completed_commit = git(root, "rev-parse", "HEAD")
    monkeypatch.setattr("scripts.execute_notebooks.subprocess.run", real_run)
    assert push_publication(root, "results/retry") == completed_commit
    assert git(remote, "rev-parse", "results/retry") == completed_commit
