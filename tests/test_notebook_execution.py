"""Notebook resume, publication safety, and provenance regression contracts."""

import json
import shutil
import time
from pathlib import Path

import nbformat
import pytest

from jigsaw_rules.review import public_evidence
from jigsaw_rules.runtime import Progress, atomic_json, digest
from scripts.execute_notebooks import (
    execute_inprocess,
    execute_one,
    publish_notebook,
    select_notebooks,
    source_hash,
    validate_execution,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path):
    for directory in ("src", "scripts", "reports", "notebooks", "configs"):
        shutil.copytree(ROOT / directory, tmp_path / directory)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(ROOT / name, tmp_path / name)
    return tmp_path


def tiny_notebook(project, source="print('verified')"):
    path = project / "notebooks/03_saved_results.ipynb"
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])
    nbformat.write(nb, path)
    return path


def test_source_hash_ignores_outputs_but_not_narrative():
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print(1)")])
    before = source_hash(nb)
    nb.cells[0].execution_count = 1
    nb.cells[0].outputs = [nbformat.v4.new_output("stream", name="stdout", text="1")]
    assert source_hash(nb) == before
    nb.cells.append(nbformat.v4.new_markdown_cell("Changed interpretation"))
    assert source_hash(nb) != before


@pytest.mark.parametrize("name", ["../secrets.ipynb", "kaggle/submission.ipynb", "99", "*.ipynb"])
def test_selector_rejects_unapproved_paths(project, name):
    with pytest.raises(ValueError, match="Select an existing"):
        select_notebooks(project, [name], "public")


def test_selector_accepts_numbers_and_deduplicates(project):
    paths = select_notebooks(project, ["03", "03_saved_results.ipynb"], "public")
    assert [p.name for p in paths] == ["03_saved_results.ipynb"]
    assert len(select_notebooks(project, None, "public")) == 5
    assert len(select_notebooks(ROOT, None, "synthetic")) == 1


def test_publication_rejects_synthetic_before_accessing_files(tmp_path):
    with pytest.raises(ValueError, match="Only public"):
        publish_notebook(tmp_path / "source", tmp_path / "executed", "", "synthetic")


@pytest.mark.parametrize("kind", ["baseline", "semantic"])
def test_public_evidence_is_real_and_provenance_linked(kind):
    evidence = public_evidence(ROOT, kind)
    assert evidence["data_kind"] == "competition"
    assert evidence["training_rows"] == 2029
    assert "not OOF recomputation" in evidence["verification"]
    assert len(evidence["results"]) == 4


def test_public_evidence_detects_changed_artifact(project):
    (project / "reports/baseline/results.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum mismatch"):
        public_evidence(project, "baseline")


def test_new_model_report_invalidates_notebook_cache(project):
    from scripts.execute_notebooks import execution_contract

    path = tiny_notebook(project)
    nb = nbformat.read(path, as_version=4)
    before = execution_contract(project, project, nb, "public", "inprocess")
    folder = project / "reports/model_validation"
    folder.mkdir(exist_ok=True)
    (folder / "results.json").write_text("[]")
    after = execution_contract(project, project, nb, "public", "inprocess")
    assert before != after
    assert "reports/model_validation/results.json" in after["inputs"]


def test_public_evidence_rejects_unsafe_manifest_path(project):
    path = project / "reports/baseline/metadata.json"
    metadata = json.loads(path.read_text())
    metadata["files"]["../../private.json"] = "a" * 64
    atomic_json(path, metadata)
    with pytest.raises(ValueError, match="contract"):
        public_evidence(project, "baseline")


def test_public_evidence_rejects_synthetic_identity(project):
    path = project / "reports/baseline/provenance.json"
    provenance = json.loads(path.read_text())
    provenance["config"]["synthetic"] = True
    atomic_json(path, provenance)
    manifest = path.parent / "metadata.json"
    metadata = json.loads(manifest.read_text())
    metadata["files"]["provenance.json"] = digest(path)
    atomic_json(manifest, metadata)
    with pytest.raises(ValueError, match="identity"):
        public_evidence(project, "baseline")


def test_resume_reuses_intact_outputs_and_input_change_invalidates(project, monkeypatch):
    path = tiny_notebook(project)
    with Progress(project / "logs/test.jsonl", "test") as log:
        first = execute_one(project, project, path, "public", "inprocess", log)
        before = first.read_bytes()

        def forbidden(*args, **kwargs):
            raise RuntimeError("execution attempted")

        monkeypatch.setattr("scripts.execute_notebooks.execute_inprocess", forbidden)
        second = execute_one(project, project, path, "public", "inprocess", log)
        assert first == second and second.read_bytes() == before
        # Even a changed byte in an input changes the execution contract.
        with (project / "reports/baseline/results.json").open("a") as stream:
            stream.write("\n")
        with pytest.raises(RuntimeError, match="execution attempted"):
            execute_one(project, project, path, "public", "inprocess", log)


def test_corrupt_notebook_output_is_recomputed(project, monkeypatch):
    path = tiny_notebook(project)
    with Progress(project / "logs/test.jsonl", "test") as log:
        cached = execute_one(project, project, path, "public", "inprocess", log)
        cached.write_text("corrupted")
        calls = []

        def tracked(*args, **kwargs):
            calls.append(1)
            return execute_inprocess(*args, **kwargs)

        monkeypatch.setattr("scripts.execute_notebooks.execute_inprocess", tracked)
        restored = execute_one(project, project, path, "public", "inprocess", log)
        validate_execution(nbformat.read(restored, as_version=4))
        assert calls == [1]


def test_failure_preserves_canonical_and_does_not_commit(project):
    path = tiny_notebook(project, "raise RuntimeError('intentional test failure')")
    before = path.read_bytes()
    with Progress(project / "logs/test.jsonl", "test") as log:
        with pytest.raises(RuntimeError, match="intentional test failure"):
            execute_one(project, project, path, "public", "inprocess", log)
    assert path.read_bytes() == before
    assert not list((project / "runs/notebook_execution").glob("*/03_saved_results/complete.json"))


def test_publication_is_atomic_and_rejects_concurrent_edits(project):
    path = tiny_notebook(project)
    original = digest(path)
    with Progress(project / "logs/test.jsonl", "test") as log:
        cached = execute_one(project, project, path, "public", "inprocess", log)
    publish_notebook(path, cached, original, "public")
    assert path.read_bytes() == cached.read_bytes()
    current = digest(path)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed during execution"):
        publish_notebook(path, cached, current, "public")


def test_unexecuted_cells_errors_and_stderr_fail_closed():
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print(1)")])
    with pytest.raises(ValueError, match="unexecuted"):
        validate_execution(nb)
    nb.cells[0].execution_count = 1
    nb.cells[0].outputs = [nbformat.v4.new_output("stream", name="stderr", text="Warning")]
    with pytest.raises(ValueError, match="stderr"):
        validate_execution(nb)
    nb.cells[0].outputs = [
        nbformat.v4.new_output("error", ename="Error", evalue="test", traceback=[])
    ]
    with pytest.raises(ValueError, match="execution error"):
        validate_execution(nb)


def test_heartbeat_has_stage_and_total_times(tmp_path):
    start = time.monotonic() - 2
    path = tmp_path / "events.jsonl"
    with Progress(path, "timing", heartbeat_seconds=0.01, total_started=start):
        time.sleep(0.04)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert any(row["event"] == "heartbeat" for row in records)
    assert all(row["total_elapsed_seconds"] >= row["stage_elapsed_seconds"] + 1 for row in records)
    assert all(row["timestamp"].endswith("+00:00") for row in records)


def test_nested_progress_inherits_total_clock_and_restores_context(tmp_path):
    path = tmp_path / "nested.jsonl"
    with Progress(path, "parent", total_started=time.monotonic() - 2) as parent:
        with Progress(path, "child") as child:
            assert child.total_started == parent.total_started
    with Progress(path, "independent") as independent:
        assert independent.total_started == independent.started


def test_execution_tags_invalidate_source_hash():
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print(1)")])
    before = source_hash(nb)
    nb.cells[0].metadata["tags"] = ["skip-execution"]
    assert source_hash(nb) != before
