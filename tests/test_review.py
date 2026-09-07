import json

import pytest

from jigsaw_rules.data import synthetic
from jigsaw_rules.model import LexicalClassifier
from jigsaw_rules.pipeline import run_baseline
from jigsaw_rules.review import review_run, select_run, verified_stage
from jigsaw_rules.runtime import atomic_json, digest


@pytest.fixture
def experiment(tmp_path):
    synthetic(tmp_path / "data/raw")
    run = run_baseline(tmp_path)
    return tmp_path, run


def test_synthetic_never_selected_as_competition(experiment):
    root, run = experiment
    with pytest.raises(FileNotFoundError, match="competition-data"):
        select_run(root, None)
    with pytest.raises(FileNotFoundError, match="competition-data"):
        review_run(root, run.name)


def test_saved_review_never_trains_or_changes_run(experiment, monkeypatch):
    root, run = experiment
    before = {p: digest(p) for p in run.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError("Review must never train")

    monkeypatch.setattr(LexicalClassifier, "fit", forbidden)
    evidence = review_run(root, run.name, allow_synthetic=True)
    assert evidence["data_kind"] == "synthetic"
    assert evidence["run_id"] == run.name
    assert len(evidence["results"]) == 4
    assert before == {p: digest(p) for p in run.rglob("*") if p.is_file()}
    assert json.loads((root / "reports/private/results.json").read_text()) == evidence


def test_corrupt_completed_result_is_rejected(experiment):
    root, run = experiment
    (run / "review/results.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum mismatch"):
        review_run(root, run.name, allow_synthetic=True)


def test_metrics_are_recomputed_from_saved_predictions(experiment):
    root, run = experiment
    path = run / "review/results.json"
    records = json.loads(path.read_text())
    records[0]["metrics"]["rule_macro_auc"] = 0.123
    atomic_json(path, records)
    marker_path = run / "review/complete.json"
    marker = json.loads(marker_path.read_text())
    marker["files"]["results.json"] = digest(path)
    atomic_json(marker_path, marker)
    with pytest.raises(ValueError, match="do not match saved OOF"):
        review_run(root, run.name, allow_synthetic=True)


def test_inconsistent_data_identity_is_rejected(experiment):
    root, run = experiment
    atomic_json(run / "status.json", {"status": "completed", "synthetic": False})
    with pytest.raises(ValueError, match="identity"):
        review_run(root, run.name)


def test_run_identifier_cannot_escape_project(tmp_path):
    with pytest.raises(ValueError, match="fingerprint"):
        select_run(tmp_path, "../../other")


def test_checkpoint_paths_cannot_escape_stage(tmp_path):
    atomic_json(tmp_path / "review/complete.json", {"files": {"../outside": "0" * 64}})
    with pytest.raises(ValueError, match="Unsafe"):
        verified_stage(tmp_path, "review")
