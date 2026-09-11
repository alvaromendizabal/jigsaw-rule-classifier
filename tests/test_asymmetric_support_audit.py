from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.audit_asymmetric_support import run_audit


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    rule = "No advertising"
    plan = {
        "schema": 1,
        "protocol": "new_rule_supplied_support_adaptation",
        "folds": [
            {
                "rule": rule,
                "audit": {},
                "training": [
                    {"body": "positive near", "rule": rule, "rule_violation": 1, "repeat": 2},
                    {"body": "positive far", "rule": rule, "rule_violation": 1, "repeat": 2},
                    {"body": "negative near", "rule": rule, "rule_violation": 0, "repeat": 2},
                    {"body": "negative far", "rule": rule, "rule_violation": 0, "repeat": 2},
                ],
                "queries": [
                    {"row_id": 10, "body": "query one", "rule": rule},
                    {"row_id": 11, "body": "query two", "rule": rule},
                ],
            }
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    representations = tmp_path / "repr"
    representations.mkdir()
    vector_path = representations / "fold_0.npz"
    np.savez_compressed(
        vector_path,
        train_adapted_vectors=np.array(
            [[1.0, 0.0], [-1.0, 0.0], [0.8, 0.2], [-0.9, 0.1]], dtype=np.float32
        ),
        query_adapted_vectors=np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32),
        query_row_ids=np.array([10, 11], dtype=np.int64),
    )
    config = {
        "schema": 1,
        "name": "asymmetric_cached_support_selection",
        "evidence_scope": "software-test",
        "base_commit": "synthetic",
        "plan": {"sha256": _sha(plan_path)},
        "representations": [
            {"name": "fold_0.npz", "bytes": vector_path.stat().st_size, "sha256": _sha(vector_path)}
        ],
        "dimensions": 2,
        "expected_counts": [{"training_rows": 4, "query_rows": 2, "same_rule_supports": 4}],
        "selection": {
            "method": "nearest_positive_farthest_negative",
            "positive_examples": 1,
            "negative_examples": 1,
            "epsilon": 1e-12,
            "batch_size": 1,
        },
        "audit": {
            "no_automatic_inference_or_promotion": True,
            "stop_if_any_policy_matches_lexical": True,
        },
        "runtime": {"max_seconds": 30, "heartbeat_seconds": 1, "cpu_threads": 1},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, plan_path, representations


def test_audit_runs_target_free_and_saves_private_pairs(tmp_path):
    config, plan, representations = _fixture(tmp_path)
    output = tmp_path / "output"
    summary = run_audit(config, plan, representations, output, 30, 1)
    assert summary["total_queries"] == 2
    assert summary["query_targets_read"] is False
    assert summary["prediction_arrays_read"] is False
    assert summary["new_model_calls"] == 0
    run = output / summary["run_id"]
    assert (run / "private/selected_pairs.json").is_file()
    assert (run / "complete.json").is_file()


def test_completed_audit_replays_without_selection(tmp_path, monkeypatch):
    config, plan, representations = _fixture(tmp_path)
    output = tmp_path / "output"
    first = run_audit(config, plan, representations, output, 30, 1)
    import jigsaw_rules.asymmetric_support as candidate

    def forbidden(*args, **kwargs):
        raise AssertionError("completed replay recomputed selection")

    monkeypatch.setattr(candidate, "select_batch", forbidden)
    second = run_audit(config, plan, representations, output, 30, 1)
    assert second == first


def test_corrupted_completed_payload_is_rejected(tmp_path):
    config, plan, representations = _fixture(tmp_path)
    output = tmp_path / "output"
    summary = run_audit(config, plan, representations, output, 30, 1)
    audit = output / summary["run_id"] / "audit.json"
    audit.write_text("{}", encoding="utf-8")
    try:
        run_audit(config, plan, representations, output, 30, 1)
    except ValueError as exc:
        assert "corruption" in str(exc).lower()
    else:
        raise AssertionError("corrupted completed audit should be rejected")
