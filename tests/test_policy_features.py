from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from scripts import policy_features as pf
from scripts import run_policy_features as run


def test_training_only_normalized_rule_map():
    m = pf.PolicyMap().fit(["  No ADS ", "no ads", "No advice"])
    assert m.rules_ == ("no ads", "no advice")
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    block, info = m.augment(x, ["NO ADS", "unseen", "NO ADVICE"], mode="condition")
    np.testing.assert_array_equal(block.toarray(), [[1, 2, 0, 0], [0, 0, 0, 0], [0, 0, 5, 6]])
    assert info["unknown_rows"] == 1
    assert m.rules_ == ("no ads", "no advice")


@pytest.mark.parametrize("mode", ["copy", "condition"])
def test_unknown_policy_is_shared_only(mode):
    m = pf.PolicyMap().fit(["a", "b"])
    out, _ = m.augment([[9, 2]], ["never seen"], mode=mode)
    assert out.nnz == 0 and out.shape == (1, 4)


@pytest.mark.parametrize("shape", [(1, 1), (10, 4), (20, 11), (7, 0)])
def test_matched_shape_sparsity_and_row_norm(shape):
    rng = np.random.default_rng(15)
    x = rng.normal(size=shape)
    x[x < 0.5] = 0
    rules = ["a" if i % 2 else "b" for i in range(shape[0])]
    m = pf.PolicyMap().fit(["a", "b"])
    blocks, _ = pf.matched_blocks(x, rules, m)
    a, b = blocks["condition"], blocks["copy"]
    assert a.shape == b.shape and a.nnz == b.nnz
    np.testing.assert_allclose(a.multiply(a).sum(1), b.multiply(b).sum(1), atol=1e-12)


def test_query_data_cannot_change_training_map_or_transform():
    m = pf.PolicyMap().fit(["a", "b"])
    before, _ = m.augment([[1, 2], [3, 4]], ["a", "b"], mode="condition")
    m.augment([[999, -555]], ["unseen"], mode="condition")
    after, _ = m.augment([[1, 2], [3, 4]], ["a", "b"], mode="condition")
    np.testing.assert_array_equal(before.toarray(), after.toarray())


@pytest.mark.parametrize("bad", [[], [None], [""], ["  "], [np.nan], [1]])
def test_bad_rule_values_stop(bad):
    with pytest.raises(ValueError, match="rule"):
        pf.PolicyMap().fit(bad)


def test_unknown_mode_unfitted_misaligned_nonfinite():
    with pytest.raises(ValueError, match="fit"):
        pf.PolicyMap().augment([[1]], ["a"], mode="copy")
    m = pf.PolicyMap().fit(["a"])
    with pytest.raises(ValueError, match="mode"):
        m.augment([[1]], ["a"], mode="other")
    with pytest.raises(ValueError, match="aligned"):
        m.augment([[1], [2]], ["a"], mode="copy")
    with pytest.raises(ValueError, match="finite"):
        m.augment([[np.nan]], ["a"], mode="copy")


def test_duplicate_sparse_coordinates_canonicalized():
    x = sparse.coo_matrix(([1.0, 2.0, 0.0], ([0, 0, 0], [0, 0, 1])), shape=(1, 2))
    block, info = pf.PolicyMap().fit(["a"]).augment(x, ["a"], mode="condition")
    assert info["nonzeros"] == 1
    np.testing.assert_array_equal(block.toarray(), [[3, 0]])


def test_authored_opposite_policy_effect_is_representable():
    # Deliberately constructed mechanics test, not a real-world moderation label.
    x = np.array([[1.0], [-1.0], [1.0], [-1.0]] * 8)
    rules = ["a", "a", "b", "b"] * 8
    y = np.array([1, 0, 0, 1] * 8)
    mapper = pf.PolicyMap().fit(rules)
    blocks, _ = pf.matched_blocks(x, rules, mapper)
    actual = sparse.hstack([x, blocks["condition"]], format="csr")
    control = sparse.hstack([x, blocks["copy"]], format="csr")
    model = LogisticRegression(C=1, solver="liblinear").fit(actual, y)
    baseline = LogisticRegression(C=1, solver="liblinear").fit(control, y)
    assert roc_auc_score(y, model.predict_proba(actual)[:, 1]) == 1
    assert roc_auc_score(y, baseline.predict_proba(control)[:, 1]) == 0.5


def test_prediction_parity_accepts_both_coefficient_shapes(tmp_path):
    ids, x = np.array([1, 2]), np.zeros((2, 3))
    for coef in (np.zeros(3), np.zeros((1, 3))):
        p = tmp_path / "p.npz"
        np.savez(
            p,
            row_ids=ids,
            probability=np.array([0.5, 0.5]),
            coefficients=coef,
            intercept=np.array([0.0]),
        )
        assert run.load_prediction(p, ids, x)[0].tolist() == [0.5, 0.5]
        with pytest.raises(ValueError, match="order"):
            run.load_prediction(p, ids[::-1], x)
    with pytest.raises(ValueError, match="dimension"):
        run.load_prediction(p, ids, np.zeros((2, 4)))
    np.savez(
        p,
        row_ids=ids,
        probability=np.array([0.7, 0.7]),
        coefficients=np.zeros(3),
        intercept=np.array([0.0]),
    )
    with pytest.raises(ValueError, match="parity"):
        run.load_prediction(p, ids, x)


def test_no_silent_promotion_of_secondary_candidate():
    metrics, contrasts, pooled = [], [], []
    for name in run.VARIANTS:
        for fold in range(2):
            metrics.append(
                {"fold": fold, "variant": name, "auc": 0.99 if name == "condition_both" else 0.5}
            )
        pooled.append({"variant": name, "ranked_pooled_auc": 0.5})
    for ref in (run.ANCHOR, "copy_lexical"):
        contrasts.append(
            {
                "candidate": "condition_lexical",
                "reference": ref,
                "delta_auc": 0.0,
                "simultaneous_low": -0.01,
            }
        )
    config = {"primary": "condition_lexical", "minimum_macro_delta": 0.003}
    assert run.decide(metrics, contrasts, pooled, config)["decision"] == "DO_NOT_PROMOTE_PRIMARY"


def test_config_registered_and_no_gpu():
    config = json.loads((Path(__file__).parents[1] / run.CONFIG).read_text())
    assert config["primary"] == "condition_lexical"
    assert config["new_fits"] == 2 * len(run.NEW) == 12
    assert config["cached_control_fits"] == 6
    assert config["automatic_gpu_authorization"] is False
    assert len(run.CONTRASTS) == 13


@pytest.fixture(scope="module")
def study_template(tmp_path_factory):
    from scripts import run_behavioral_features as old
    from scripts import run_relational_features as second
    from tests.test_behavioral_features import authored_frame

    root = tmp_path_factory.mktemp("round3_integration")
    source = Path(__file__).parents[1]
    for name in set(run.SOURCE_PATHS):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    (root / "data/raw").mkdir(parents=True)
    frame = authored_frame()
    frame.to_csv(root / "data/raw/train.csv", index=False)
    spec = json.loads((root / old.CONFIG).read_text())
    spec.update(
        word_features=120,
        char_features=150,
        feature_budget_per_family=8,
        bootstrap_replicates=40,
        max_seconds=120,
    )
    prior = old.run_study(root, test_frame=frame, test_config=spec)
    config2 = json.loads((root / second.CONFIG).read_text())
    config2.update(
        prior_run_id=prior["run_id"],
        prior_results_sha256=run.digest(root / "reports/behavioral_features/results.json"),
        train_sha256=run.digest(root / "data/raw/train.csv"),
        query_counts=[48, 48],
        bootstrap_replicates=40,
        max_seconds=120,
    )
    (root / second.CONFIG).write_text(json.dumps(config2, indent=2) + "\n")
    two = second.run_study(root)
    config3 = json.loads((root / run.CONFIG).read_text())
    config3.update(
        round2_run_id=two["run_id"],
        round2_results_sha256=run.digest(root / "reports/relational_features/results.json"),
        train_sha256=run.digest(root / "data/raw/train.csv"),
        query_counts=[48, 48],
        bootstrap_replicates=40,
        max_seconds=120,
    )
    (root / run.CONFIG).write_text(json.dumps(config3, indent=2) + "\n")
    return root


@pytest.fixture
def study_root(study_template, tmp_path):
    return Path(shutil.copytree(study_template, tmp_path / "project"))


def test_complete_study_reuses_controls_and_preserves_old_work(study_root):
    before = {
        str(p): run.digest(p)
        for group in (
            "runs/behavioral_features",
            "reports/behavioral_features",
            "runs/relational_features",
            "reports/relational_features",
        )
        for p in (study_root / group).rglob("*")
        if p.is_file()
    }
    result = run.run_study(study_root)
    assert result["new_fits"] == 12 and result["reused_prior_controls"] == 6
    assert result["control_design_parity"] is True and not result["automatic_gpu_authorization"]
    assert all(run.digest(Path(name)) == digest for name, digest in before.items())
    assert len(run.figures(result)) == 8
    assert all(b["row_norm_parity"] for b in result["blocks"])
    assert all(c["query_text_in_training"] == 0 for c in result["cohorts"])


def test_interrupted_stages_are_reused(study_root):
    with pytest.raises(TimeoutError, match="authored interruption"):
        run.run_study(study_root, max_new_fits=3)
    result = run.run_study(study_root)
    assert result["new_fits"] == 9 and result["reused_new_fits"] == 3


def test_finished_replay_never_calls_fit(study_root, monkeypatch):
    first = run.run_study(study_root)

    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected fit on replay")

    monkeypatch.setattr(run, "fit_candidate", forbidden)
    assert run.run_study(study_root) == first


def test_query_targets_not_passed_to_features(study_root, monkeypatch):
    original = run.prepare_fold

    def checked(fold, *args, **kwargs):
        assert all(set(q) == {"body", "rule", "row_id"} for q in fold["queries"])
        return original(fold, *args, **kwargs)

    monkeypatch.setattr(run, "prepare_fold", checked)
    run.run_study(study_root)


def test_changed_reference_source_stops_before_any_fit(study_root, monkeypatch):
    (study_root / "scripts/relational_features.py").write_text("# edited\n")
    monkeypatch.setattr(run, "fit_candidate", lambda *a, **k: pytest.fail("fit called"))
    with pytest.raises(ValueError, match="source changed"):
        run.run_study(study_root)


def test_corrupt_round2_predictions_stop(study_root):
    config = json.loads((study_root / run.CONFIG).read_text())
    p = study_root / "runs/relational_features" / config["round2_run_id"]
    (p / "fold_1/add_act_roles/predictions.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        run.run_study(study_root)


def test_missing_round2_checkpoint_does_not_trigger_refit(study_root):
    config = json.loads((study_root / run.CONFIG).read_text())
    p = study_root / "runs/relational_features" / config["round2_run_id"]
    (p / "fold_1/add_act_roles/complete.json").unlink()
    with pytest.raises(ValueError, match="checkpoint missing"):
        run.run_study(study_root)


def test_completed_candidate_corruption_stops(study_root):
    result = run.run_study(study_root)
    p = study_root / run.PRIVATE / result["run_id"] / "fold_0/condition_lexical/predictions.npz"
    p.write_bytes(p.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        run.run_study(study_root)


def test_public_export_has_no_private_arrays_or_raw_data(study_root, tmp_path):
    import zipfile

    run.run_study(study_root)
    p = run.export_return(study_root, tmp_path / "out.zip")
    with zipfile.ZipFile(p) as z:
        assert not any(n.endswith(".npz") or n.startswith("data/") for n in z.namelist())
        for n, sha in json.loads(z.read("SHA256SUMS.json")).items():
            assert run.hashlib.sha256(z.read(n)).hexdigest() == sha


def test_sparse_input_not_mutated():
    x = sparse.csr_matrix(([1.0, 2.0], [0, 0], [0, 2]), shape=(1, 2))
    old_data, old_indices = x.data.copy(), x.indices.copy()
    pf.PolicyMap().fit(["a"]).augment(x, ["a"], mode="copy")
    np.testing.assert_array_equal(x.data, old_data)
    np.testing.assert_array_equal(x.indices, old_indices)
