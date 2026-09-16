from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from scripts import evidence_features as ef
from scripts import run_evidence_features as run
from tests.test_policy_features import study_template as policy_template  # noqa: F401


def matrix():
    return sparse.csr_matrix([[4.0, 0, 1], [2, 0, 1], [0, 3, 1], [0, 1, 1], [1, 2, 0], [3, 0, 0]])


def transformer():
    return ef.RuleEvidence().fit(
        matrix(), [1, 1, 0, 0, 0, 1], ["ads", "ads", "ads", "ads", "law", "law"]
    )


def test_manual_log_ratio_matches_incidence_formula():
    x = sparse.csr_matrix([[3, 1, 0], [0, 1, 2], [2, 0, 0]])
    y = np.array([1, 0, 1])
    w = np.array([1.0, 1.0, 2.0])
    # Positive incidence [3,1,0], negative [0,1,1], then add-one smoothing.
    want = np.log(np.array([4, 2, 1]) / 7) - np.log(np.array([1, 2, 2]) / 5)
    np.testing.assert_allclose(ef.log_ratio(x, y, w, 1), want)


def test_evidence_uses_incidence_not_repeated_token_magnitude():
    x = matrix()
    modified = x.copy()
    modified.data *= np.arange(1, len(x.data) + 1)
    args = ([1, 1, 0, 0, 0, 1], ["ads"] * 4 + ["law"] * 2)
    a = ef.RuleEvidence().fit(x, *args)
    b = ef.RuleEvidence().fit(modified, *args)
    np.testing.assert_allclose(a.weights("law", "rule"), b.weights("law", "rule"))


@pytest.mark.parametrize("mode", ["global", "rule", "permuted", "unshrunk"])
def test_norm_sparsity_shape_preserved(mode):
    x = matrix()
    before = x.copy()
    actual, info = transformer().transform(x, ["ads"] * 4 + ["law"] * 2, mode=mode)
    np.testing.assert_allclose(ef.row_norms(actual), ef.row_norms(x), atol=1e-12)
    assert actual.shape == x.shape and actual.nnz == x.nnz
    assert info["row_norm_max_abs_error"] < 1e-12
    assert (x != before).nnz == 0


def test_all_one_weights_leave_matrix_unchanged():
    np.testing.assert_allclose(
        ef.reweight_preserving_norm(matrix(), np.ones(3)).toarray(), matrix().toarray()
    )


def test_empty_rows_remain_empty():
    x = sparse.csr_matrix([[0, 0, 0], [1, 2, 0]])
    out, _ = transformer().transform(x, ["ads", "law"], mode="rule")
    assert out[0].nnz == 0
    np.testing.assert_allclose(ef.row_norms(x), ef.row_norms(out))


def test_permutation_preserves_multiset_and_is_deterministic():
    model = transformer()
    ordinary = model.weights("ads", "rule")
    permuted = model.weights("ads", "permuted")
    np.testing.assert_array_equal(np.sort(ordinary), np.sort(permuted))
    np.testing.assert_array_equal(permuted, model.weights("ads", "permuted"))


def test_query_content_cannot_change_fitted_weights():
    model = transformer()
    before = model.weights("ads", "rule")
    model.transform(matrix() * 900, ["new"] * 6, mode="rule")
    np.testing.assert_array_equal(before, model.weights("ads", "rule"))
    assert set(model.local_) == {"ads", "law"}


def test_shrinkage_uses_unique_class_counts_not_repeat_weight():
    model = ef.RuleEvidence().fit(matrix(), [1, 1, 0, 0, 0, 1], ["a"] * 6, [100] * 6)
    assert model.reliability_["a"] == 3 / 67
    model2 = ef.RuleEvidence().fit(matrix(), [1, 1, 0, 0, 0, 1], ["a"] * 6)
    assert model2.reliability_ == model.reliability_


def test_missing_policy_class_falls_back_to_global():
    model = ef.RuleEvidence().fit(matrix(), [1, 1, 0, 0, 0, 1], ["a", "a", "b", "b", "b", "a"])
    assert model.reliability_["a"] == 0
    np.testing.assert_allclose(model.weights("a", "rule"), model.weights("a", "global"))


def test_unknown_policy_uses_global_without_refitting():
    model = transformer()
    np.testing.assert_allclose(model.weights("other", "rule"), model.weights("ads", "global"))


def test_class_sign_flip_does_not_create_a_fake_new_feature():
    y = np.array([1, 1, 0, 0, 0, 1])
    rules = ["ads"] * 4 + ["law"] * 2
    a = ef.RuleEvidence().fit(matrix(), y, rules)
    b = ef.RuleEvidence().fit(matrix(), 1 - y, rules)
    np.testing.assert_allclose(a.weights("ads", "rule"), b.weights("ads", "rule"))


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_invalid_parameters_rejected(value):
    for key in ("alpha", "shrinkage", "cap"):
        with pytest.raises(ValueError, match="positive"):
            ef.RuleEvidence(**{key: value})


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
def test_bad_matrix_rejected(value):
    with pytest.raises(ValueError):
        ef.checked_matrix([[value, 1]])


@pytest.mark.parametrize("labels", [[1] * 6, [0, 1], [1, 1, 0, 0, 0, 0.5]])
def test_invalid_training_labels_rejected(labels):
    with pytest.raises(ValueError):
        ef.RuleEvidence().fit(matrix(), labels, ["a"] * 6)


@pytest.mark.parametrize("weights", [[0] * 6, [1, 2], [np.nan] * 6, [-1] * 6])
def test_invalid_sample_weights_rejected(weights):
    with pytest.raises(ValueError):
        ef.RuleEvidence().fit(matrix(), [1, 1, 0, 0, 0, 1], ["a"] * 6, weights)


def test_unknown_mode_and_unfitted_transform_stop():
    with pytest.raises(ValueError, match="fit"):
        ef.RuleEvidence().weights("ads", "rule")
    with pytest.raises(ValueError, match="fit"):
        ef.RuleEvidence().transform(matrix(), ["a"] * 6, mode="rule")
    with pytest.raises(ValueError, match="mode"):
        transformer().weights("ads", "tuned")
    with pytest.raises(ValueError, match="dimensions"):
        transformer().transform([[1, 2]], ["ads"], mode="rule")


def test_duplicate_sparse_input_does_not_double_count_incidence():
    x = sparse.coo_matrix(([1, 2, 3], ([0, 0, 1], [0, 0, 1])), shape=(2, 2))
    clean = ef.checked_matrix(x)
    assert clean.nnz == 2
    assert clean[0, 0] == 3


def test_profiles_do_not_export_vocabulary():
    rows = transformer().profile(family="word", fold=0)
    assert len(rows) == 6
    assert all(1 <= row["minimum"] <= row["maximum"] <= 4 for row in rows)
    assert not any("tokens" in r or "vocabulary" in r for r in rows)


def test_preregistered_counts_and_primary():
    cfg = json.loads((Path(__file__).parents[1] / run.CONFIG).read_text())
    assert cfg["new_fits"] == 2 * len(run.NEW) == 12
    assert cfg["cached_control_fits"] == 2 * len(run.CONTROLS) == 10
    assert len(run.CONTRASTS) == 12
    assert cfg["primary"] == "rule_both"
    assert not cfg["automatic_gpu_authorization"]


def test_secondary_winner_cannot_replace_primary():
    metrics, pooled, comparisons = [], [], []
    for name in run.VARIANTS:
        for fold in range(2):
            metrics.append({"variant": name, "fold": fold, "auc": 0.5})
        pooled.append({"variant": name, "ranked_pooled_auc": 0.5})
    for ref in (run.ANCHOR, "permuted_both"):
        comparisons.append(
            {"candidate": "rule_both", "reference": ref, "delta_auc": 0, "simultaneous_low": -0.1}
        )
    cfg = {"primary": "rule_both", "minimum_macro_delta": 0.003}
    assert run.decide(metrics, comparisons, pooled, cfg)["decision"] == "DO_NOT_PROMOTE_PRIMARY"


@pytest.fixture(scope="module")
def study_template(policy_template, tmp_path_factory):  # noqa: F811
    root = Path(shutil.copytree(policy_template, tmp_path_factory.mktemp("r4") / "project"))
    source = Path(__file__).parents[1]
    for name in run.SOURCE_PATHS:
        if name not in run.policy.SOURCE_PATHS:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, target)
    third = run.policy.run_study(root)
    config = json.loads((root / run.CONFIG).read_text())
    config.update(
        round3_run_id=third["run_id"],
        round3_results_sha256=run.digest(root / run.policy.PUBLIC / "results.json"),
        train_sha256=run.digest(root / "data/raw/train.csv"),
        query_counts=[48, 48],
        bootstrap_replicates=40,
        max_seconds=120,
    )
    (root / run.CONFIG).write_text(json.dumps(config, indent=2) + "\n")
    return root


@pytest.fixture
def study_root(study_template, tmp_path):
    return Path(shutil.copytree(study_template, tmp_path / "project"))


def test_full_run_parity_norms_and_prior_bytes_preserved(study_root):
    prior_groups = [
        "runs/behavioral_features",
        "reports/behavioral_features",
        "runs/relational_features",
        "reports/relational_features",
        "runs/policy_features",
        "reports/policy_features",
    ]
    before = {
        str(p): run.digest(p)
        for group in prior_groups
        for p in (study_root / group).rglob("*")
        if p.is_file()
    }
    result = run.run_study(study_root)
    assert result["new_fits"] == 12 and result["reused_prior_controls"] == 10
    assert result["control_design_parity"] and result["kaggle_score"] is None
    assert all(run.digest(Path(name)) == sha for name, sha in before.items())
    assert len(run.figures(result)) == 8
    assert len(result["weight_stability"]) == 4
    for check in result["norm_checks"]:
        assert all(f["query_error"] < 1e-10 for f in check["families"])


def test_interruption_reuses_completed_stages(study_root):
    with pytest.raises(TimeoutError, match="authored interruption"):
        run.run_study(study_root, max_new_fits=3)
    result = run.run_study(study_root)
    assert result["new_fits"] == 9 and result["reused_new_fits"] == 3


def test_replay_never_refits(study_root, monkeypatch):
    first = run.run_study(study_root)

    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected fit")

    monkeypatch.setattr(run, "fit_candidate", forbidden)
    assert run.run_study(study_root) == first


def test_query_targets_not_sent_to_features(study_root, monkeypatch):
    original = run.prepare_fold

    def checked(fold, *args, **kwargs):
        assert all(set(q) == {"row_id", "body", "rule"} for q in fold["queries"])
        return original(fold, *args, **kwargs)

    monkeypatch.setattr(run, "prepare_fold", checked)
    run.run_study(study_root)


def test_old_source_change_blocks_new_fits(study_root, monkeypatch):
    (study_root / "scripts/policy_features.py").write_text("# edited\n")
    monkeypatch.setattr(run, "fit_candidate", lambda *a, **k: pytest.fail("unexpected fit"))
    with pytest.raises(ValueError, match="source changed"):
        run.run_study(study_root)


def test_old_prediction_corruption_stops(study_root):
    config = json.loads((study_root / run.CONFIG).read_text())
    p = study_root / run.policy.PRIVATE / config["round3_run_id"]
    (p / "fold_0/condition_lexical/predictions.npz").write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        run.run_study(study_root)


def test_missing_old_checkpoint_never_triggers_refit(study_root):
    config = json.loads((study_root / run.CONFIG).read_text())
    p = study_root / run.policy.PRIVATE / config["round3_run_id"]
    (p / "fold_0/condition_lexical/complete.json").unlink()
    with pytest.raises(ValueError, match="checkpoint missing"):
        run.run_study(study_root)


def test_current_candidate_corruption_stops(study_root):
    result = run.run_study(study_root)
    p = study_root / run.PRIVATE / result["run_id"] / "fold_0/rule_both/predictions.npz"
    p.write_bytes(p.read_bytes() + b"broken")
    with pytest.raises(ValueError, match="checksum"):
        run.run_study(study_root)


def test_existing_public_result_with_other_identity_preserved(study_root):
    result = run.run_study(study_root)
    p = study_root / run.PUBLIC / "results.json"
    before = p.read_bytes()
    (study_root / run.PRIVATE / result["run_id"] / "finished.json").unlink()
    config_path = study_root / run.CONFIG
    config = json.loads(config_path.read_text())
    config["bootstrap_replicates"] += 1
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="different completed"):
        run.run_study(study_root)
    assert p.read_bytes() == before


def test_export_contains_aggregates_not_private_arrays(study_root, tmp_path):
    import zipfile

    run.run_study(study_root)
    path = run.export_return(study_root, tmp_path / "return.zip")
    with zipfile.ZipFile(path) as z:
        assert not any(
            n.endswith((".npz", ".joblib")) or n.startswith("data/") for n in z.namelist()
        )
        for name, sha in json.loads(z.read("SHA256SUMS.json")).items():
            assert run.hashlib.sha256(z.read(name)).hexdigest() == sha
