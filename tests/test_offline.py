"""Offline delivery integrity, input boundaries and interruption recovery."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import jigsaw_rules.offline as offline
from jigsaw_rules.data import EXAMPLES, synthetic
from jigsaw_rules.runtime import atomic_json, digest


class FixedCandidate:
    """Software fixture only; real accepted-model parity is a separate integration gate."""

    def predict(self, frame, vectors):
        return np.full(len(frame), 0.4)


@pytest.fixture
def inputs(tmp_path):
    directory = tmp_path / "data"
    synthetic(directory)
    return directory / "test.csv"


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    joblib.dump(FixedCandidate(), root / "candidate.joblib")
    repository = Path(__file__).resolve().parents[1]
    (root / "configs").mkdir()
    (root / "configs/semantic.json").write_bytes(
        (repository / "configs/semantic.json").read_bytes()
    )
    record = {
        "schema": 1,
        "kind": "accepted_postcompetition_route",
        "packages": offline.software_versions(),
        "sources": {p.name: digest(p) for p in Path(offline.__file__).parent.glob("*.py")},
        "candidate_sha256": digest(root / "candidate.joblib"),
        "files": {
            "candidate.joblib": digest(root / "candidate.joblib"),
            "configs/semantic.json": digest(root / "configs/semantic.json"),
        },
    }
    atomic_json(root / "bundle.json", record)
    return root, digest(root / "bundle.json")


def test_external_manifest_pin_precedes_deserialization(bundle, monkeypatch):
    root, checksum = bundle
    monkeypatch.setattr(joblib, "load", lambda *a, **k: pytest.fail("Must not load modified model"))
    (root / "candidate.joblib").write_bytes(b"changed serialized object")
    with pytest.raises(ValueError, match="checksum"):
        offline.OfflineModel(root, checksum, root.parent / "cache")


@pytest.mark.parametrize("kind", ["manifest", "source", "packages", "path"])
def test_changed_contract_is_rejected(bundle, kind):
    root, checksum = bundle
    manifest = json.loads((root / "bundle.json").read_text())
    if kind == "source":
        manifest["sources"]["offline.py"] = "0" * 64
    elif kind == "packages":
        manifest["packages"]["numpy"] = "different"
    elif kind == "path":
        manifest["files"]["../outside"] = "0" * 64
    else:
        manifest["kind"] = "different"
    atomic_json(root / "bundle.json", manifest)
    if kind != "manifest":
        checksum = digest(root / "bundle.json")
    with pytest.raises(ValueError):
        offline.verify_bundle(root, checksum)


@pytest.mark.parametrize("column", [*EXAMPLES, "body", "rule"])
def test_missing_required_text_rejected_before_model_loading(bundle, inputs, column, monkeypatch):
    root, checksum = bundle
    model = offline.OfflineModel(root, checksum, root.parent / "cache")
    monkeypatch.setattr(joblib, "load", lambda *a, **k: pytest.fail("Unexpected model load"))
    monkeypatch.setattr(model.encoder, "encode", lambda *a, **k: pytest.fail("Unexpected encoding"))
    frame = pd.read_csv(inputs)
    frame.loc[0, column] = " "
    with pytest.raises(ValueError, match="Empty or non-text"):
        model.predict(frame)


def test_target_bearing_input_rejected(bundle, inputs):
    root, checksum = bundle
    frame = pd.read_csv(inputs)
    frame["rule_violation"] = 0
    frame.to_csv(inputs, index=False)
    with pytest.raises(ValueError, match="target labels"):
        offline.predict_file(root, checksum, inputs, root.parent / "out", root.parent / "cache")


def test_completed_batches_survive_interruption_and_are_not_repredicted(
    bundle, inputs, monkeypatch
):
    root, checksum = bundle
    calls = []

    def predict(self, frame):
        calls.append(frame.row_id.tolist())
        if len(calls) == 2:
            raise RuntimeError("Interrupted active batch")
        return pd.DataFrame({"row_id": frame.row_id.to_numpy(), "rule_violation": 0.4}), {}

    monkeypatch.setattr(offline.OfflineModel, "predict", predict)
    args = (root, checksum, inputs, root.parent / "out", root.parent / "cache")
    with pytest.raises(RuntimeError, match="Interrupted"):
        offline.predict_file(*args, batch_rows=3)
    assert not (args[3] / "predictions.csv").exists()
    marker = next((args[4] / "predictions").glob("*/batch_000000000/complete.json"))
    before = marker.stat().st_mtime_ns
    result = offline.predict_file(*args, batch_rows=3)
    assert len(pd.read_csv(result)) == 8
    assert calls.count([1000, 1001, 1002]) == 1
    assert marker.stat().st_mtime_ns == before
    monkeypatch.setattr(offline.OfflineModel, "predict", lambda *a, **k: pytest.fail("Not reused"))
    first = result.read_bytes()
    assert offline.predict_file(*args, batch_rows=3).read_bytes() == first
    damaged = marker.parent / "predictions.csv"
    damaged.write_bytes(b"corrupt batch")
    with pytest.raises(ValueError, match="checksum"):
        offline.predict_file(*args, batch_rows=3)


def test_candidate_cannot_change_after_lazy_verification(bundle, inputs):
    root, checksum = bundle
    model = offline.OfflineModel(root, checksum, root.parent / "cache")
    (root / "candidate.joblib").write_bytes(b"replaced after verification")
    with pytest.raises(ValueError, match="changed before loading"):
        model.predict_vectors(pd.read_csv(inputs), np.ones((8, 5, 8)))
