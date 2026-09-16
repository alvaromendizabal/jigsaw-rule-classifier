"""New evaluation tests: synthetic data and temporary directories only."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.metrics import roc_auc_score

from scripts import feature_combination_evaluation as run


def test_fixed_plan_covers_every_group_both_directions():
    p = run.plan()
    assert len(p) == 49
    assert [sum(r["batch"] == b for r in p) * 2 for b in (1, 2, 3)] == [34, 34, 30]
    for g in run.GROUPS:
        a = next(r for r in p if r["name"] == "add_" + g)
        d = next(r for r in p if r["name"] == "without_" + g)
        assert a["groups"] == [g]
        assert set(d["groups"]) == set(run.GROUPS) - {g}


def test_exact_duplicates_and_constants_training_only():
    x = np.array([[1, 1, 5, 0], [2, 2, 5, 1], [3, 3, 5, 4]], float)
    q = np.array([[10, 20, 99, 0]], float)
    a, _, info = run.dense_transform(x, q)
    b, _, other = run.dense_transform(x, -q)
    np.testing.assert_array_equal(a, b)
    assert info == other
    assert info["keep"] == [0, 3]
    assert x[0].tolist() == [1, 1, 5, 0]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_dense_stops(bad):
    with pytest.raises(ValueError):
        run.dense_transform([[0, 1], [1, bad]], [[0, 0]])


def test_different_column_width_stops():
    with pytest.raises(ValueError, match="alignment"):
        run.dense_transform([[0, 1], [1, 0]], [[0]])


def test_all_constants_stop():
    with pytest.raises(ValueError, match="variable"):
        run.dense_transform(np.ones((4, 2)), np.ones((2, 2)))


def test_fit_deterministic_and_does_not_modify_input():
    rng = np.random.default_rng(4)
    x = sparse.csr_matrix(rng.normal(size=(50, 5)))
    q = x[:5].copy()
    y = np.arange(50) % 2
    original = x.toarray().copy()
    a = run.fit_model(x, y, q, np.ones(50), {"C": 1.0})
    b = run.fit_model(x, y, q, np.ones(50), {"C": 1.0})
    np.testing.assert_array_equal(a["score"], b["score"])
    np.testing.assert_array_equal(x.toarray(), original)
    assert np.allclose(a["probability"], run.expit(a["score"]))


@pytest.mark.parametrize("bad", [np.zeros(20), np.full(20, np.nan), -np.ones(20)])
def test_bad_weights_stop(bad):
    with pytest.raises(ValueError, match="weights"):
        run.fit_model(sparse.eye(20), np.arange(20) % 2, sparse.eye(20), bad, {"C": 1})


def test_original_count_is_not_classifier_fit(monkeypatch):
    from jigsaw_rules.data import EXAMPLES

    f = pd.DataFrame({"body": ["one two", "three four"], "rule": ["rule here"] * 2})
    for col in EXAMPLES:
        f[col] = ["example hello", "example hello"]
    monkeypatch.setattr(run.LogisticRegression, "fit", lambda *a, **kw: pytest.fail("fit"))
    r = run.first_submission_count(f)
    assert r["total_columns"] == r["word_unigram_bigram_columns"] + 8
    assert r["classifier_fits"] == 0 and r["total_columns"] < 40008


def test_saved_bank_requires_shape_and_checksum(tmp_path):
    p = tmp_path / "bank.npz"
    np.savez(p, training=np.zeros((5, 3)), query=np.zeros((4, 3)))
    run.read_arrays(p, 5, 4, 3)
    with pytest.raises(ValueError, match="alignment"):
        run.read_arrays(p, 5, 4, 4)
    with pytest.raises(ValueError, match="changed"):
        run.check_file(p, "0" * 64)


def test_missing_bank_does_not_build_any_features(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        run.bank(
            tmp_path, {"example": {"run_id": "a"}}, "example", 0, "feature_bank", "x.npz", 2, 5, 4
        )


def test_metric_ties_and_labels():
    y = np.array([0, 0, 1, 1])
    s = np.array([0, 1, 1, 2])
    assert run.metrics(y, s)["auc"] == roc_auc_score(y, s)
    assert run.metrics(y, np.zeros(4))["auc"] == 0.5


def test_scope_copy_norms_match_without_query_fitting():
    text = [
        'You should sue. "not advice"',
        "Should I sue my landlord?",
        "You should ask a lawyer.",
        "not legal advice ask a lawyer",
    ] * 6
    t = pd.DataFrame(
        {
            "body": text,
            "rule": ["No legal advice"] * 24,
            "rule_violation": np.arange(24) % 2,
            "repeat": np.ones(24),
        }
    )
    q = pd.DataFrame({"body": ['You should sue "not advice"'], "rule": ["No legal advice"]})
    a, info = run.lexical_blocks(t, q)
    b, _ = run.lexical_blocks(t, q.assign(body="unseenworddifferent no context"))
    for mode in a:
        np.testing.assert_allclose(a[mode][0].toarray(), b[mode][0].toarray())
    np.testing.assert_allclose(a["scope"][0].power(2).sum(1), a["copy"][0].power(2).sum(1))
    assert info["word_columns"] > 0


def test_export_excludes_private_arrays(tmp_path):
    root = tmp_path / "root"
    (root / run.PUBLIC).mkdir(parents=True)
    (root / run.PUBLIC / "results.json").write_text('{"status":"test"}')
    (root / run.PRIVATE).mkdir(parents=True)
    (root / run.PRIVATE / "private.npz").write_bytes(b"secret")
    (root / "data").mkdir()
    (root / "data/private.csv").write_text("raw private")
    out = run.export(root, tmp_path / "out.zip")
    with zipfile.ZipFile(out) as z:
        assert not any(n.endswith(".npz") or n.startswith("data/") for n in z.namelist())
        for n, h in json.loads(z.read("SHA256SUMS.json")).items():
            assert hashlib.sha256(z.read(n)).hexdigest() == h


def test_no_automatic_promotion_or_ensemble_search():
    source = Path(run.__file__).read_text()
    assert "GridSearchCV(" not in source and 'automatic_gpu_authorization": True' not in source
    assert "NONE_AUTOMATIC_DEVELOPMENT_ONLY" in source


def test_stage_integrity_and_recovery(tmp_path):
    key = {"run": "test"}
    assert run.stage_read(tmp_path, key) is None
    run.stage_write(tmp_path, {"test.json": b"{}"}, key)
    assert run.stage_read(tmp_path, key)
    (tmp_path / "test.json").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        run.stage_read(tmp_path, key)


def test_source_has_no_long_lines():
    assert all(len(line) <= 100 for line in Path(run.__file__).read_text().splitlines())
