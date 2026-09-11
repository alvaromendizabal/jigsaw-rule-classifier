import numpy as np
import pytest

from jigsaw_rules.support_selection_csls import (
    SelectorConfig,
    csls_scores,
    promotion_diagnostics,
    select_fold,
)


def fold() -> dict:
    return {
        "rule": "No ads",
        "training": [
            {"body": "sell shoes", "rule": "No ads", "rule_violation": 1, "repeat": 2},
            {"body": "buy now", "rule": "No ads", "rule_violation": 1, "repeat": 2},
            {
                "body": "discussion only",
                "rule": "No ads",
                "rule_violation": 0,
                "repeat": 2,
            },
            {"body": "news link", "rule": "No ads", "rule_violation": 0, "repeat": 2},
            {
                "body": "other rule row",
                "rule": "No insults",
                "rule_violation": 1,
                "repeat": 1,
            },
        ],
        "queries": [
            {"row_id": 10, "body": "special offer", "rule": "No ads"},
            {"row_id": 11, "body": "helpful article", "rule": "No ads"},
        ],
        "audit": {},
    }


def test_csls_shape_and_finite() -> None:
    query = np.array([[1.0, 0.0], [0.0, 1.0]])
    support = np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
    output = csls_scores(query, support, k=2)
    assert output.shape == (2, 3)
    assert np.isfinite(output).all()


def test_selection_is_deterministic_and_target_free() -> None:
    sample = fold()
    train = np.eye(5, 3, k=0)[:, :3]
    train[3] = [0, 1, 0]
    train[4] = [1, 1, 0]
    query = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    first = select_fold(sample, train, query, SelectorConfig(k=2))
    second = select_fold(sample, train, query, SelectorConfig(k=2))
    assert first == second
    assert first["queries"] == 2


def test_alignment_mismatch_stops() -> None:
    with pytest.raises(ValueError, match="alignment"):
        select_fold(fold(), np.ones((4, 3)), np.ones((2, 3)))


def test_query_leakage_stops() -> None:
    sample = fold()
    sample["training"][0]["body"] = "special offer"
    with pytest.raises(ValueError, match="leaked"):
        select_fold(sample, np.ones((5, 3)), np.ones((2, 3)))


def test_duplicate_pair_stops() -> None:
    sample = fold()
    sample["training"][1]["body"] = "sell shoes"
    with pytest.raises(ValueError, match="duplicate"):
        select_fold(sample, np.ones((5, 3)), np.ones((2, 3)))


def test_zero_query_stops() -> None:
    with pytest.raises(ValueError, match="zero-norm query"):
        select_fold(fold(), np.ones((5, 3)), np.zeros((2, 3)))


def test_missing_nonzero_class_stops() -> None:
    train = np.ones((5, 3))
    train[0] = 0
    train[1] = 0
    with pytest.raises(ValueError, match="support class"):
        select_fold(fold(), train, np.ones((2, 3)))


def test_csls_hubness_transform_is_well_defined() -> None:
    query = np.array([[1.0, 0.0], [0.8, 0.6], [0.8, -0.6]], dtype=float)
    query = query / np.linalg.norm(query, axis=1, keepdims=True)
    support = np.array([[1.0, 0.0], [0.98, 0.2]], dtype=float)
    support = support / np.linalg.norm(support, axis=1, keepdims=True)
    raw = query @ support.T
    adjusted = csls_scores(query, support, k=2)
    assert int(np.argmax(raw[0])) == 0
    assert adjusted.shape == raw.shape


def test_promotion_diagnostic_never_authorizes_gpu() -> None:
    result = {
        "changed_pairs": 1,
        "raw": {
            "positive": {"maximum_reuse_fraction": 0.8},
            "negative": {"maximum_reuse_fraction": 0.7},
        },
        "csls": {
            "positive": {"maximum_reuse_fraction": 0.4},
            "negative": {"maximum_reuse_fraction": 0.3},
        },
    }
    gate = promotion_diagnostics(result, 0.5)
    assert gate["eligible_for_blinded_relevance_review"] is True
    assert gate["gpu_inference_authorized"] is False
