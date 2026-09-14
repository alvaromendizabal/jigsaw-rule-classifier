from __future__ import annotations

import ast
import hashlib
import json
import shutil
import tokenize
import zipfile
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import GroupKFold

from jigsaw_rules.data import normalize
from jigsaw_rules.runtime import atomic_json, digest
from scripts import matched_support_features as feat
from scripts import run_local_support_features as previous
from scripts import run_matched_support_features as run


def inputs(n=42, d=12):
    rng = np.random.default_rng(71)
    x = rng.normal(size=(n, d))
    y = np.arange(n) % 2
    x[:, 0] += 0.5 * (2 * y - 1)
    q = rng.normal(size=(8, d))
    texts = np.asarray([f"reference {i:03}" for i in range(n)])
    return x, y, texts, q


def test_shape_and_immutability():
    x, y, text, q = inputs()
    before = x.copy(), y.copy(), q.copy()
    out, stats = feat.paired_geometry(x, y, text, q, seed=21)
    assert len(feat.GLOBAL) == 4 and len(feat.LOCAL) == 8
    assert set(out) == set(feat.MODES)
    for mode in feat.MODES:
        assert out[mode].shape == (len(q), 12)
        assert np.isfinite(out[mode]).all()
        assert stats[mode]["maximum_endpoint_reuse"] == 1
    for a, b in zip(before, (x, y, q), strict=True):
        np.testing.assert_array_equal(a, b)


def test_exact_assignment_objective():
    x, y, text, _ = inputs()
    r = feat.unit(x)
    expected = r[y == 1] @ r[y == 0].T
    a, b = linear_sum_assignment(-expected)
    p, n, stats = feat.matched_endpoints(x, y, text)
    assert np.sum(p * n) == pytest.approx(expected[a, b].sum())
    assert len(p) == len(n) == stats["pairs"] == 21


def test_random_pair_control_retains_identical_endpoints():
    x, y, text, q = inputs()
    _, stats = feat.paired_geometry(x, y, text, q, seed=1)
    assert stats["matched"]["pairs"] == stats["repaired"]["pairs"]
    assert stats["matched"]["unused_reference_rows"] == 0
    assert stats["repaired"]["same_selected_endpoints"]
    assert stats["matched"]["mean_pair_cosine"] >= stats["repaired"]["mean_pair_cosine"]


def test_unbalanced_labels_report_unused_rows():
    x, y, text, q = inputs()
    y[0:10] = 0
    _, stats = feat.paired_geometry(x, y, text, q, seed=2)
    n = min((y == 0).sum(), (y == 1).sum())
    assert stats["matched"]["pairs"] == n
    assert stats["matched"]["unused_reference_rows"] == len(y) - 2 * n


def test_query_batch_and_order_invariance():
    x, y, text, q = inputs()
    a, _ = feat.paired_geometry(x, y, text, q, seed=3)
    reverse, _ = feat.paired_geometry(x, y, text, q[::-1], seed=3)
    singles = [feat.paired_geometry(x, y, text, row[None], seed=3)[0] for row in q]
    for m in feat.MODES:
        np.testing.assert_allclose(reverse[m], a[m][::-1], atol=1e-12)
        np.testing.assert_allclose(np.vstack([row[m] for row in singles]), a[m], atol=1e-12)


def test_reference_order_does_not_change_pair_features_or_null_controls():
    x, y, text, q = inputs()
    a, _ = feat.paired_geometry(x, y, text, q, seed=4)
    b, _ = feat.paired_geometry(x[::-1], y[::-1], text[::-1], q, seed=4)
    for m in feat.MODES:
        np.testing.assert_allclose(a[m], b[m], atol=1e-12)


def test_positive_scaling_and_common_rotation_invariance():
    x, y, text, q = inputs()
    rotation, _ = np.linalg.qr(np.random.default_rng(10).normal(size=(x.shape[1], x.shape[1])))
    a, _ = feat.paired_geometry(x, y, text, q, seed=2)
    b, _ = feat.paired_geometry((x @ rotation) * 3, y, text, (q @ rotation) * 4, seed=2)
    for m in feat.MODES:
        np.testing.assert_allclose(a[m], b[m], atol=1e-11)


def test_sign_orientation_control_keeps_matching_diagnostics():
    x, y, text, q = inputs()
    a, stats = feat.paired_geometry(x, y, text, q, seed=8)
    assert stats["matched"] == stats["orientation"]
    assert not np.allclose(a["matched"], a["orientation"])
    assert not np.allclose(a["matched"], a["unnormalized"])


def test_authored_common_topic_cancels_in_pair_direction():
    positive = feat.unit(np.array([[3.0, 1.0, 0.0], [0.0, 1.0, 3.0]]))
    negative = feat.unit(np.array([[3.0, -1.0, 0.0], [0.0, -1.0, 3.0]]))
    queries = feat.unit(np.array([[3.0, 0.0, 0.0], [3.0, 1.0, 0.0], [3.0, -1.0, 0.0]]))
    values, _ = feat.summarize_pairs(positive, negative, queries)
    assert values[0, 0] == pytest.approx(0)
    assert values[1, 0] > 0 and values[2, 0] < 0
    # This is a geometry test, not invented moderation ground truth.


def test_class_flip_changes_signed_evidence():
    x, y, text, q = inputs()
    p, n, _ = feat.matched_endpoints(x, y, text)
    a, _ = feat.summarize_pairs(p, n, q)
    b, _ = feat.summarize_pairs(n, p, q)
    signed = [0, 2, 4, 5, 6, 8, 10, 11]
    np.testing.assert_allclose(a[:, signed], -b[:, signed], atol=1e-12)
    np.testing.assert_allclose(a[:, [1, 7, 9]], b[:, [1, 7, 9]], atol=1e-12)


@pytest.mark.parametrize("issue", ["zero", "nan", "inf", "dimension", "label", "text"])
def test_bad_inputs_rejected(issue):
    x, y, text, q = inputs()
    if issue == "zero":
        x[0] = 0
    elif issue == "nan":
        x[0, 0] = np.nan
    elif issue == "inf":
        q[0, 0] = np.inf
    elif issue == "dimension":
        q = q[:, :-1]
    elif issue == "label":
        y[:] = 0
    else:
        text[0] = text[1]
    with pytest.raises(ValueError):
        feat.paired_geometry(x, y, text, q, seed=1)


def test_degenerate_pair_handling():
    p = np.array([[1.0, 0.0], [0.0, 1.0]])
    n = np.array([[1.0, 0.0], [1.0, 0.0]])
    out, stats = feat.summarize_pairs(p, n, np.array([[1.0, 0.0]]))
    assert stats["degenerate_pairs_excluded"] == 1 and np.isfinite(out).all()
    with pytest.raises(ValueError, match="nondegenerate"):
        feat.summarize_pairs(p, p, p)


def test_crossfit_own_labels_do_not_enter_features():
    x, y, text, q = inputs()
    rules = ["same rule"] * len(x)
    a, _ = feat.cross_fitted(x, y, text, rules, q, seed=9)
    _, held = next(GroupKFold(n_splits=3).split(x, groups=text))
    changed = y.copy()
    changed[held] = 1 - changed[held]
    b, _ = feat.cross_fitted(x, changed, text, rules, q, seed=9)
    for m in feat.MODES:
        np.testing.assert_allclose(a[m][0][held], b[m][0][held], atol=1e-12)


def test_crossfit_query_values_do_not_affect_training_features():
    x, y, text, q = inputs()
    rules = ["same rule"] * len(x)
    a, _ = feat.cross_fitted(x, y, text, rules, q, seed=11)
    b, _ = feat.cross_fitted(x, y, text, rules, -q, seed=11)
    for m in feat.MODES:
        np.testing.assert_array_equal(a[m][0], b[m][0])


def test_cross_rule_references_forbidden():
    x, y, text, q = inputs()
    with pytest.raises(ValueError, match="same rule"):
        feat.cross_fitted(x, y, text, ["a", "b"] * (len(x) // 2), q, seed=1)


def make_project(root):
    from tests.test_local_support_features import make_project as make_prior

    root = make_prior(root)
    result = previous.run_study(root)
    source = Path(__file__).parents[1]
    for name in (*run.SOURCES, run.NOTEBOOK):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    spec = json.loads((root / run.CONFIG).read_text())
    spec.update(
        round6_run_id=result["run_id"],
        round6_results_sha256=digest(root / previous.PUBLIC / "results.json"),
        bootstrap_replicates=40,
    )
    atomic_json(root / run.CONFIG, spec)
    return root


@pytest.fixture(scope="module")
def study_template(tmp_path_factory):
    return make_project(tmp_path_factory.mktemp("paired") / "project")


@pytest.fixture
def project(study_template, tmp_path):
    return Path(shutil.copytree(study_template, tmp_path / "project"))


def test_end_to_end_and_prior_artifacts_unchanged(project):
    paths = [
        p
        for group in (
            "data",
            previous.PUBLIC,
            previous.PRIVATE,
            "runs/feature_value_audit",
            "reports/feature_value_audit",
        )
        for p in (project / group).rglob("*")
        if p.is_file()
    ]
    before = {str(p): digest(p) for p in paths}
    result = run.run_study(project)
    assert result["new_fits"] == 12 and result["reused_prior_controls"] == 6
    assert result["control_design_parity"] and len(result["comparisons"]) == 13
    assert len(result["metrics"]) == 18
    assert result["new_neural_inference"] == 0 and result["kaggle_score"] is None
    assert not result["automatic_gpu_authorization"]
    assert len(run.figures(result)) == 8
    assert all(digest(Path(p)) == h for p, h in before.items())


def test_interrupted_three_fits_reused(project):
    with pytest.raises(TimeoutError, match="authored interruption"):
        run.run_study(project, max_new_fits=3)
    result = run.run_study(project)
    assert result["new_fits"] == 9 and result["reused_new_fits"] == 3


def test_complete_replay_never_fits_or_builds_features(project, monkeypatch):
    first = run.run_study(project)

    def forbidden(*args, **kwargs):
        pytest.fail("completed work recomputed")

    monkeypatch.setattr(run, "fit_readout", forbidden)
    monkeypatch.setattr(run, "feature_bank", forbidden)
    assert run.run_study(project) == first
    assert json.loads((project / run.PRIVATE / "last_invocation.json").read_text())["new_fits"] == 0


@pytest.mark.parametrize("kind", ["prediction", "bank", "public", "missing_bank"])
def test_completed_corruption_or_missing_stage_stops(project, kind, monkeypatch):
    result = run.run_study(project)
    private = project / run.PRIVATE / result["run_id"]
    paths = {
        "prediction": private / "fold_0/paired_all/predictions.npz",
        "bank": private / "fold_0/pair_bank/matched.npz",
        "public": project / run.PUBLIC / "results.json",
        "missing_bank": private / "fold_0/pair_bank/complete.json",
    }
    path = paths[kind]
    if kind == "missing_bank":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b"corrupt")
    monkeypatch.setattr(run, "fit_readout", lambda *a, **k: pytest.fail("unexpected fit"))
    with pytest.raises(ValueError):
        run.run_study(project)


@pytest.mark.parametrize("kind", ["source", "raw", "prior_public", "prior_model", "prior_feature"])
def test_prior_changes_stop_before_any_fit(project, kind, monkeypatch):
    spec = json.loads((project / run.CONFIG).read_text())
    folder = project / previous.PRIVATE / spec["round6_run_id"]
    path = {
        "source": project / "scripts/local_support_features.py",
        "raw": project / "data/raw/train.csv",
        "prior_public": project / previous.PUBLIC / "results.json",
        "prior_model": folder / "fold_1/answer_only/predictions.npz",
        "prior_feature": folder / "fold_1/features_frozen/features.npz",
    }[kind]
    path.write_bytes(path.read_bytes() + b"modified")
    monkeypatch.setattr(run, "fit_readout", lambda *a, **k: pytest.fail("unexpected fit"))
    with pytest.raises(ValueError):
        run.run_study(project)


def test_all_controls_checked_before_new_fit(project, monkeypatch):
    original = run.prepare_controls
    checked = []

    def verify(*args, **kwargs):
        value = original(*args, **kwargs)
        assert len(value[0]) == 6
        checked.append(True)
        return value

    fitter = run.fit_readout

    def fit(*args, **kwargs):
        assert checked == [True]
        return fitter(*args, **kwargs)

    monkeypatch.setattr(run, "prepare_controls", verify)
    monkeypatch.setattr(run, "fit_readout", fit)
    run.run_study(project)


def test_feature_inputs_have_no_query_targets(project, monkeypatch):
    original = run.feature_bank

    def bank(bundle, *args, **kwargs):
        assert set(bundle["query"].columns) == {"body", "rule", "row_id"}
        assert len(set(bundle["train"].rule.map(normalize))) == 1
        return original(bundle, *args, **kwargs)

    monkeypatch.setattr(run, "feature_bank", bank)
    run.run_study(project)


def test_secondary_winner_not_promoted():
    metrics = [
        {"fold": f, "variant": n, "auc": 0.99 if n == "paired_local" else 0.5}
        for f in range(2)
        for n in run.VARIANTS
    ]
    contrasts = [
        {"candidate": a, "reference": b, "delta_auc": 0, "simultaneous_low": -0.1}
        for a, b, _ in run.CONTRASTS
    ]
    pooled = [{"variant": n, "ranked_pooled_auc": 0.5} for n in run.VARIANTS]
    spec = {"primary": "paired_all", "minimum_macro_delta": 0.003}
    assert run.decide(metrics, contrasts, pooled, spec)["decision"] == "DO_NOT_PROMOTE_PRIMARY"


def test_export_has_no_private_data(project, tmp_path):
    run.run_study(project)
    path = run.export(project, tmp_path / "return.zip")
    with zipfile.ZipFile(path) as z:
        assert not any(n.startswith("data/") or n.endswith((".npz", ".npy")) for n in z.namelist())
        hashes = json.loads(z.read("SHA256SUMS.json"))
        assert len(hashes) == len(z.namelist()) - 1
        assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in hashes.items())


def test_notebook_display_only_and_eight_cells():
    import nbformat

    book = nbformat.read(Path(__file__).parents[1] / run.NOTEBOOK, as_version=4)
    cells = [c for c in book.cells if c.cell_type == "code"]
    assert len(cells) == 8
    assert not any("run_study(" in c.source or "fit(" in c.source for c in cells)


def test_registered_config_and_memory_budget():
    spec = json.loads((Path(__file__).parents[1] / run.CONFIG).read_text())
    run.validate_config(spec)
    spec["primary"] = "paired_local"
    with pytest.raises(ValueError):
        run.validate_config(spec)
    x, y, text, q = inputs()
    with pytest.raises(ValueError, match="memory"):
        feat.paired_geometry(
            np.ones((4001, 12)), np.arange(4001) % 2, [str(i) for i in range(4001)], q, seed=1
        )


def test_static_source_limits():
    root = Path(__file__).parents[1]
    for name in run.SOURCES:
        if not name.endswith(".py"):
            continue
        source = (root / name).read_text()
        assert max(map(len, source.splitlines())) <= 100, name
        ast.parse(source, feature_version=(3, 12))
        with (root / name).open("rb") as stream:
            for t in tokenize.tokenize(stream.readline):
                if t.type == tokenize.STRING and t.start[0] == t.end[0]:
                    assert len(t.string) <= 85, (name, t.start)
