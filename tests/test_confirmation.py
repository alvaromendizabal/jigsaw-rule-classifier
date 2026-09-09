"""The confirmation boundary must fail before labels or incompatible artifacts are used."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from jigsaw_rules import confirmation
from jigsaw_rules.data import normalize
from jigsaw_rules.diagnostics import support_scores
from jigsaw_rules.embeddings import content_key
from jigsaw_rules.runtime import atomic_bytes, atomic_json, stage
from tests.test_expanded import cohort


def test_preregistered_protocol_cannot_silently_change(tmp_path):
    root = Path(__file__).resolve().parents[1]
    original = (root / "configs/confirmation.json").read_bytes()
    atomic_bytes(tmp_path / "configs/confirmation.json", original)
    assert confirmation.load_protocol(tmp_path)["eligible_rows"] == 43509
    atomic_bytes(tmp_path / "configs/confirmation.json", original + b" ")
    with pytest.raises(ValueError, match="preregistration"):
        confirmation.load_protocol(tmp_path)


def test_completed_confirmation_stage_refuses_corruption_instead_of_recomputing(tmp_path):
    path = stage(tmp_path, "predictions", lambda p: atomic_json(p / "scores.json", [0.2]))
    (path / "scores.json").write_text("[0.8]")
    called = []
    with pytest.raises(ValueError):
        confirmation.strict_stage(tmp_path, "predictions", lambda p: called.append(p))
    assert not called


def test_prediction_controls_use_frozen_route_and_reject_targets():
    frame = cohort().iloc[:8].drop(columns="rule_violation")
    vectors = np.random.default_rng(8).normal(size=(len(frame), 5, 12)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
    known_rule = normalize(frame.iloc[0].rule)
    known = frame.rule.map(normalize).eq(known_rule).to_numpy()
    pipeline = SimpleNamespace(
        policies_=(known_rule,),
        predict=lambda f, v: np.full(len(f), 0.73),
    )
    candidate = SimpleNamespace(familiar=pipeline, predict=lambda f, v: np.full(len(f), 0.61))
    reference = SimpleNamespace(predict=lambda f: np.full(len(f), 0.42))
    result = confirmation.predict_batch(candidate, reference, frame, vectors)
    assert np.array_equal(result.familiar, known)
    np.testing.assert_allclose(result.uncalibrated_route[known], 0.73)
    np.testing.assert_array_equal(
        result.uncalibrated_route[~known], support_scores(vectors)["qwen_centroid"][~known]
    )
    np.testing.assert_allclose(result.candidate, 0.61)
    with pytest.raises(ValueError, match="target labels"):
        confirmation.predict_batch(candidate, reference, frame.assign(rule_violation=1), vectors)


def test_input_identity_detects_reordering_even_if_stage_is_recommitted(tmp_path, monkeypatch):
    frame = cohort().iloc[:8].drop(columns="rule_violation").reset_index(drop=True)
    ids = frame.row_id.tolist()
    policies = {str(i): rule for i, rule in enumerate(frame.rule.unique())}
    inverse = {value: key for key, value in policies.items()}
    assignments = pd.DataFrame(
        {
            "row_id": ids,
            "rule_id": frame.rule.map(inverse),
            "Usage": "Private",
            "role": "confirmation",
            "body_sha256": frame.body.map(normalize).map(
                lambda t: hashlib.sha256(t.encode()).hexdigest()
            ),
            "self_support_match": False,
        }
    )
    # Use nonnumeric policy IDs so CSV parsing preserves the real assignment schema.
    assignments["rule_id"] = assignments.rule_id.map(lambda x: "policy_" + x)
    policies = {"policy_" + k: v for k, v in policies.items()}
    monkeypatch.setattr(
        confirmation,
        "load_protocol",
        lambda root: {
            "eligible_rows": len(frame),
            "eligible_ids_sha256": content_key(ids),
        },
    )
    monkeypatch.setattr(confirmation, "released_plan", lambda root: {"rule_text": policies})
    identity = {
        "protocol_sha256": confirmation.PROTOCOL_SHA256,
        "frame_content_key": content_key(frame.to_csv(index=False)),
        "assignment_content_key": content_key(assignments.to_csv(index=False)),
    }

    def write(path):
        atomic_bytes(path / "frame.csv", frame.to_csv(index=False).encode())
        atomic_bytes(path / "assignments.csv", assignments.to_csv(index=False).encode())
        atomic_json(path / "identity.json", identity)

    stage(tmp_path, "inputs", write)
    restored, _ = confirmation.read_inputs(tmp_path, tmp_path)
    assert restored.row_id.tolist() == ids
    path = tmp_path / "inputs"
    atomic_bytes(path / "frame.csv", frame.iloc[::-1].to_csv(index=False).encode())
    marker = json.loads((path / "complete.json").read_text())
    marker["files"]["frame.csv"] = confirmation.digest(path / "frame.csv")
    atomic_json(path / "complete.json", marker)
    with pytest.raises(ValueError, match="identities"):
        confirmation.read_inputs(tmp_path, tmp_path)
