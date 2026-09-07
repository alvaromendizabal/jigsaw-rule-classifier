import json
import threading

import pandas as pd
import pytest

from jigsaw_rules.data import synthetic
from jigsaw_rules.model import LexicalClassifier
from jigsaw_rules.pipeline import run_baseline
from jigsaw_rules.runtime import Progress, atomic_json, fingerprint, stage


def test_completed_stage_reused_and_corruption_recomputed(tmp_path):
    calls = []

    def action(dest):
        calls.append(1)
        atomic_json(dest / "value.json", {"value": 1})

    path = stage(tmp_path, "fit", action)
    stage(tmp_path, "fit", action)
    assert len(calls) == 1
    (path / "value.json").write_text("corrupt")
    stage(tmp_path, "fit", action)
    assert len(calls) == 2


def test_failure_does_not_commit_and_rerun_succeeds(tmp_path):
    def action(dest):
        (dest / "partial-output").write_text("incomplete")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        stage(tmp_path, "fit", action)
    assert not (tmp_path / "fit/complete.json").exists()
    stage(tmp_path, "fit", lambda dest: atomic_json(dest / "value.json", {"ok": True}))
    assert (tmp_path / "fit/complete.json").exists()


def test_heartbeat_and_failure_are_logged(tmp_path):
    observed = threading.Event()
    with Progress(tmp_path / "events.jsonl", "fit", heartbeat_seconds=0.01) as log:
        original = log.emit

        def emit(event, **fields):
            original(event, **fields)
            if event == "heartbeat":
                observed.set()

        log.emit = emit
        assert observed.wait(timeout=2)
    records = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert {r["event"] for r in records} >= {"started", "heartbeat", "completed"}
    assert all("timestamp" in r and "elapsed_seconds" in r for r in records)


def test_fingerprint_changes_with_inputs(tmp_path):
    synthetic(tmp_path)
    a, _ = fingerprint(tmp_path, {"seed": 1})
    b, _ = fingerprint(tmp_path, {"seed": 2})
    assert a != b
    with (tmp_path / "train.csv").open("a") as stream:
        stream.write("\n")
    c, _ = fingerprint(tmp_path, {"seed": 1})
    assert c != a


def test_end_to_end_resumes_without_refitting(tmp_path, monkeypatch):
    synthetic(tmp_path / "data/raw")
    first = run_baseline(tmp_path)
    predictions = pd.read_csv(first / "full_training/submission.csv")
    assert len(predictions) == 8
    assert "SYNTHETIC" in (first / "review/report.html").read_text()
    records = json.loads((first / "review/results.json").read_text())
    assert all(r["data_kind"] == "synthetic" and r["run_id"] == first.name for r in records)

    def should_not_fit(*args, **kwargs):
        raise AssertionError("Completed work should not be refitted")

    monkeypatch.setattr(LexicalClassifier, "fit", should_not_fit)
    assert run_baseline(tmp_path) == first
    assert '"event": "reused"' in (first / "events.jsonl").read_text()
