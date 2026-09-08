"""Exercise the documented continuation against isolated Git repos, not cloud compute."""

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = "notebooks/03_saved_results.ipynb"
LOCAL = b'{"local_execution": true}\n'
REMOTE = b'{"reviewed_public_output": true}\n'


def git(path: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        text=True,
        capture_output=True,
        check=check,
        timeout=15,
    )


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    # Ignore the caller's Git identity, hooks, signing, and configuration.
    for name in tuple(os.environ):
        if name.startswith("GIT_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    remote = tmp_path / "origin.git"
    git(tmp_path, "init", "--bare", "--initial-branch=main", str(remote))
    seed = tmp_path / "seed"
    git(tmp_path, "clone", str(remote), str(seed))
    git(seed, "config", "user.name", "Workspace test")
    git(seed, "config", "user.email", "workspace@example.invalid")
    (seed / "notebooks").mkdir()
    (seed / NOTEBOOK).write_bytes(b'{"original": true}\n')
    (seed / "kaggle").mkdir()
    (seed / "kaggle/submission.ipynb").write_bytes(b'{"original": true}\n')
    (seed / "source.py").write_text("original = True\n")
    (seed / ".gitignore").write_text(".venv/\ndata/\nruns/\nconfigs/local.json\n")
    git(seed, "add", ".")
    git(seed, "commit", "-m", "test: initialize isolated repository")
    git(seed, "push", "origin", "main")
    local = home / "projects/jigsaw-rule-classifier"
    local.parent.mkdir()
    git(tmp_path, "clone", str(remote), str(local))
    git(local, "config", "user.name", "Workspace test")
    git(local, "config", "user.email", "workspace@example.invalid")
    (seed / NOTEBOOK).write_bytes(REMOTE)
    git(seed, "commit", "-am", "test: publish reviewed notebook")
    git(seed, "push", "origin", "main")
    for name in ("data/raw/train.csv", "runs/embeddings/vectors.npy", "configs/local.json"):
        target = local / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"private-fixture-retain-exactly\x00\n")
    executable = local / ".venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> runs/test_calls.txt\n'
        'if [ "$1" = scripts/verify.py ] && [ "${JIGSAW_TEST_FAIL_VERIFY:-0}" = 1 ]; '
        "then exit 7; fi\nexit 0\n"
    )
    executable.chmod(0o755)
    return local, seed, remote


def continuation() -> subprocess.CompletedProcess:
    document = (ROOT / "START_HERE.md").read_text()
    block = re.search(
        r"<!-- workspace-update:start -->\n```bash\n(.*?)\n```\n<!-- workspace-update:end -->",
        document,
        re.DOTALL,
    )
    assert block, "The tested continuation block must remain explicitly marked"
    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", block.group(1)],
        text=True,
        capture_output=True,
        timeout=30,
    )


def calls(local: Path) -> list[str]:
    path = local / "runs/test_calls.txt"
    return path.read_text().splitlines() if path.exists() else []


def test_clean_checkout_fast_forwards_and_runs_gate_before_notebooks(workspace):
    local, seed, _ = workspace
    result = continuation()
    assert result.returncode == 0, result.stderr
    assert git(local, "rev-parse", "HEAD").stdout == git(seed, "rev-parse", "HEAD").stdout
    assert (local / NOTEBOOK).read_bytes() == REMOTE
    assert calls(local) == ["scripts/verify.py", "scripts/execute_notebooks.py"]
    assert not git(local, "stash", "list").stdout


@pytest.mark.parametrize("staged", [False, True])
def test_notebook_changes_are_retained_and_repeat_does_not_duplicate_stash(workspace, staged):
    local, _, _ = workspace
    (local / NOTEBOOK).write_bytes(LOCAL)
    if staged:
        git(local, "add", NOTEBOOK)
    result = continuation()
    assert result.returncode == 0, result.stderr
    assert git(local, "show", f"stash@{{0}}:{NOTEBOOK}").stdout.encode() == LOCAL
    assert "notebook-session-" in git(local, "stash", "list").stdout
    assert (local / NOTEBOOK).read_bytes() == REMOTE
    assert continuation().returncode == 0
    assert len(git(local, "stash", "list").stdout.splitlines()) == 1
    assert not git(local, "status", "--porcelain").stdout


@pytest.mark.parametrize("staged", [False, True])
def test_unrelated_changes_untracked_files_and_private_artifacts_remain(workspace, staged):
    local, _, _ = workspace
    (local / "source.py").write_text("local_edit = True\n")
    if staged:
        git(local, "add", "source.py")
    (local / "notebooks/local_notes.txt").write_text("do not delete\n")
    protected = [
        local / "source.py",
        local / "notebooks/local_notes.txt",
        local / "data/raw/train.csv",
        local / "runs/embeddings/vectors.npy",
        local / "configs/local.json",
    ]
    before = {path: path.read_bytes() for path in protected}
    (local / NOTEBOOK).write_bytes(LOCAL)
    result = continuation()
    assert result.returncode == 0, result.stderr
    assert {path: path.read_bytes() for path in protected} == before
    status = git(local, "status", "--porcelain", "--", "source.py").stdout
    assert status.startswith("M " if staged else " M")


@pytest.mark.parametrize("failure", ["diverged", "unreachable", "conflicting-source"])
def test_failed_pull_never_runs_verification_or_discards_notebook(workspace, failure):
    local, seed, _ = workspace
    if failure == "diverged":
        (local / "local.txt").write_text("local commit\n")
        git(local, "add", "local.txt")
        git(local, "commit", "-m", "test: preserve local commit")
    elif failure == "unreachable":
        git(local, "remote", "set-url", "origin", str(local / "missing.git"))
    else:
        (seed / "source.py").write_text("remote_edit = True\n")
        git(seed, "commit", "-am", "test: create overlapping source edit")
        git(seed, "push", "origin", "main")
        (local / "source.py").write_text("local_edit = True\n")
    (local / NOTEBOOK).write_bytes(LOCAL)
    before = git(local, "rev-parse", "HEAD").stdout
    result = continuation()
    assert result.returncode != 0
    assert calls(local) == []
    assert git(local, "rev-parse", "HEAD").stdout == before
    assert git(local, "show", f"stash@{{0}}:{NOTEBOOK}").stdout.encode() == LOCAL
    if failure == "conflicting-source":
        assert (local / "source.py").read_text() == "local_edit = True\n"


def test_wrong_branch_stops_before_stashing_or_switching(workspace):
    local, _, _ = workspace
    git(local, "switch", "-c", "feature/example")
    (local / NOTEBOOK).write_bytes(LOCAL)
    assert continuation().returncode != 0
    assert git(local, "branch", "--show-current").stdout.strip() == "feature/example"
    assert not git(local, "stash", "list").stdout
    assert (local / NOTEBOOK).read_bytes() == LOCAL
    assert calls(local) == []


def test_failed_quality_gate_never_executes_notebooks(workspace, monkeypatch):
    local, _, _ = workspace
    monkeypatch.setenv("JIGSAW_TEST_FAIL_VERIFY", "1")
    result = continuation()
    assert result.returncode == 7
    assert calls(local) == ["scripts/verify.py"]


@pytest.mark.parametrize("staged", [False, True])
def test_user_executed_submission_notebook_is_preserved(workspace, staged):
    local, seed, _ = workspace
    name = "kaggle/submission.ipynb"
    (local / name).write_bytes(LOCAL)
    if staged:
        git(local, "add", name)
    (seed / name).write_bytes(REMOTE)
    git(seed, "commit", "-am", "test: refresh canonical inference source")
    git(seed, "push", "origin", "main")
    result = continuation()
    assert result.returncode == 0, result.stderr
    assert (local / name).read_bytes() == REMOTE
    assert git(local, "show", f"stash@{{0}}:{name}").stdout.encode() == LOCAL
