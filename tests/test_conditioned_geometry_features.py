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
from sklearn.model_selection import GroupKFold

from jigsaw_rules.data import normalize
from jigsaw_rules.runtime import atomic_json, digest
from scripts import conditioned_geometry_features as feat
from scripts import run_conditioned_geometry_features as run
from scripts import run_matched_support_features as previous

OTHER_ROUND_PUBLIC = "reports/behavior_support_features"


def inputs(n=42, d=16):
    rng = np.random.default_rng(71)
    x = rng.normal(size=(n, d))
    y = np.arange(n) % 2
    x[:, 0] += 0.5 * (2 * y - 1)
    q = rng.normal(size=(8, d))
    forms = [
        "Should I hire a lawyer?",
        "You should sue the landlord.",
        "I can help with your legal claim.",
        'He wrote "You should sue" about court.',
        "Not legal advice, but you should hire a lawyer.",
        "Buy from my shop https://shop.invalid sale.",
        "See this reference https://source.invalid article.",
    ]
    texts = np.asarray([forms[i % len(forms)] + f" referenceitem{i}" for i in range(n)])
    qt = [forms[i % len(forms)] + f" newqueryitem{i}" for i in range(len(q))]
    return x, y, texts, q, qt


def cross(x, y, text, q, qt, seed=9):
    return feat.cross_fitted(x, y, text, ["same rule"] * len(x), q, qt, seed=seed)


def test_crossfit_query_values_do_not_affect_training_features():
    x, y, text, q, qt = inputs()
    a, _ = cross(x, y, text, q, qt)
    b, _ = cross(x, y, text, -500 * q, ["new comment" + str(i) for i in range(len(qt))])
    for mode in a:
        np.testing.assert_array_equal(a[mode][0], b[mode][0])


def test_self_labels_do_not_enter_own_feature_reference():
    x, y, text, q, qt = inputs()
    a, _ = cross(x, y, text, q, qt)
    groups = np.asarray([normalize(t) for t in text])
    _, held = next(GroupKFold(3).split(x, groups=groups))
    changed = y.copy()
    changed[held] = 1 - changed[held]
    b, _ = cross(x, changed, text, q, qt)
    for mode in a:
        np.testing.assert_allclose(a[mode][0][held], b[mode][0][held], atol=1e-12)


def test_query_batch_and_permutation_invariance():
    x, y, text, q, qt = inputs()
    a, _ = cross(x, y, text, q, qt)
    b, _ = cross(x, y, text, q[::-1], qt[::-1])
    one, _ = cross(x, y, text, q[:1], qt[:1])
    for mode in a:
        np.testing.assert_allclose(a[mode][1][::-1], b[mode][1], atol=1e-11)
        np.testing.assert_allclose(a[mode][1][:1], one[mode][1], atol=1e-11)


def test_reference_order_invariance():
    x, y, text, q, qt = inputs()
    a, _ = cross(x, y, text, q, qt)
    b, _ = cross(x[::-1], y[::-1], text[::-1], q, qt)
    for mode in a:
        np.testing.assert_allclose(a[mode][0][::-1], b[mode][0], atol=1e-10)
        np.testing.assert_allclose(a[mode][1], b[mode][1], atol=1e-10)


def test_inputs_not_mutated():
    x, y, text, q, qt = inputs()
    before = [a.copy() for a in (x, y, text, q)]
    cross(x, y, text, q, qt)
    for old, new in zip(before, (x, y, text, q), strict=True):
        np.testing.assert_array_equal(old, new)


@pytest.mark.parametrize(
    "fault",
    [
        "zero",
        "nan",
        "dimension",
        "labels",
        "duplicate",
        "query_overlap",
        "mixed_rules",
        "alignment",
        "empty_text",
    ],
)
def test_bad_feature_inputs_stop(fault):
    x, y, text, q, qt = inputs()
    rules = ["same rule"] * len(x)
    if fault == "zero":
        x[0] = 0
    elif fault == "nan":
        q[0, 0] = np.nan
    elif fault == "dimension":
        q = q[:, :-1]
    elif fault == "labels":
        y[:] = 0
    elif fault == "duplicate":
        text[1] = text[0]
    elif fault == "query_overlap":
        qt[0] = text[0]
    elif fault == "mixed_rules":
        rules[0] = "different rule"
    elif fault == "alignment":
        y = y[:-1]
    else:
        text[0] = ""
    with pytest.raises(ValueError):
        feat.cross_fitted(x, y, text, rules, q, qt)


def test_two_full_families_plus_centering_control():
    x, y, text, q, qt = inputs()
    out, stats = cross(x, y, text, q, qt)
    assert len(feat.NAMES) == 18
    assert set(out) == set(run.BANK_WIDTHS)
    for mode, (train, query) in out.items():
        assert train.shape == (len(x), run.BANK_WIDTHS[mode])
        assert query.shape == (len(q), run.BANK_WIDTHS[mode])
        assert np.isfinite(train).all() and np.isfinite(query).all()
    assert len(stats) == 20
    assert all(not r["query_used_in_transform_fit"] for r in stats)


def test_centering_transform_matches_formula():
    x, _, _, q, _ = inputs()
    model = feat.ReferenceTransform().fit(x)
    np.testing.assert_allclose(
        model.transform(q, "centered"), feat.unit(feat.unit(q) - feat.unit(x).mean(0)), atol=1e-12
    )


def test_deflated_vectors_orthogonal_to_removed_axes():
    x, _, _, q, _ = inputs()
    model = feat.ReferenceTransform().fit(x)
    out = model.transform(q, "deflated")
    assert np.max(np.abs(out @ model.basis_[:, : model.remove_])) < 1e-10


def test_random_axis_control_has_identical_spectrum_and_rank():
    x, _, _, q, _ = inputs(d=48)
    model = feat.ReferenceTransform().fit(x)
    np.testing.assert_allclose(model.random_basis_.T @ model.random_basis_, np.eye(32), atol=1e-12)
    assert model.rank_ == 32 and model.remove_ == 4
    assert not np.allclose(model.transform(q, "whitened"), model.transform(q, "random_whitened"))


def test_whitening_low_rank_formula_and_residual_retention():
    x, _, _, q, _ = inputs(d=48)
    model = feat.ReferenceTransform().fit(x)
    centered = feat.unit(q) - model.mean_
    expected = centered + ((centered @ model.basis_) * (model.gain_ - 1)) @ model.basis_.T
    np.testing.assert_allclose(model.transform(q, "whitened"), feat.unit(expected), atol=1e-12)
    assert model.transform(q, "whitened").shape[1] == 48
    assert np.all((model.gain_ > 0) & (model.gain_ <= 1))


def test_covariance_fit_has_no_label_or_query_parameter():
    import inspect

    assert "labels" not in inspect.signature(feat.ReferenceTransform.fit).parameters
    assert "queries" not in inspect.signature(feat.ReferenceTransform.fit).parameters


def test_basic_features_match_saved_basic_definition():
    from scripts.local_support_features import geometry

    x, y, _, q, _ = inputs()
    np.testing.assert_allclose(feat.basic_summary(x, y, q), geometry(x, y, q)[:, :9], atol=1e-12)


def test_low_dimension_clips_rank_and_direction_count():
    x, _, _, q, _ = inputs(d=4)
    model = feat.ReferenceTransform().fit(x)
    assert model.rank_ == model.remove_ == 3
    assert np.isfinite(model.transform(q, "whitened")).all()


@pytest.mark.parametrize(
    "bad",
    [{"rank": 0}, {"remove": 0}, {"shrinkage": 0}, {"shrinkage": 1}, {"shrinkage": float("nan")}],
)
def test_invalid_transform_parameters_rejected(bad):
    x, _, _, _, _ = inputs()
    with pytest.raises(ValueError):
        feat.ReferenceTransform().fit(x, **bad)


def test_degenerate_covariance_fails_clearly():
    with pytest.raises(ValueError, match="covariance"):
        feat.ReferenceTransform().fit(np.ones((12, 6)))


def test_positive_scaling_invariance():
    x, y, text, q, qt = inputs()
    a, _ = cross(x, y, text, q, qt)
    b, _ = cross(x * 3, y, text, q * 7, qt)
    for mode in a:
        np.testing.assert_allclose(a[mode][1], b[mode][1], atol=1e-10)


def test_unfitted_and_unknown_transform_rejected():
    x, _, _, q, _ = inputs()
    with pytest.raises(ValueError, match="fit"):
        feat.ReferenceTransform().transform(q, "centered")
    with pytest.raises(ValueError, match="unknown"):
        feat.ReferenceTransform().fit(x).transform(q, "unknown")


def make_project(root):
    from tests.test_matched_support_features import make_project as make_prior

    root = make_prior(root)
    result = previous.run_study(root)
    source = Path(__file__).parents[1]
    for name in (*run.SOURCES, run.NOTEBOOK):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    spec = json.loads((root / run.CONFIG).read_text())
    spec.update(
        round7_run_id=result["run_id"],
        round7_results_sha256=digest(root / previous.PUBLIC / "results.json"),
        bootstrap_replicates=40,
    )
    atomic_json(root / run.CONFIG, spec)
    return root


@pytest.fixture(scope="module")
def study_template(tmp_path_factory):
    return make_project(tmp_path_factory.mktemp("independent_round") / "project")


@pytest.fixture
def project(study_template, tmp_path):
    return Path(shutil.copytree(study_template, tmp_path / "project"))


def test_full_study_preserves_previous_scientific_artifacts(project):
    paths = [
        p
        for group in (
            "data",
            previous.PUBLIC,
            previous.PRIVATE,
            "runs/local_support_features",
            "reports/local_support_features",
        )
        for p in (project / group).rglob("*")
        if p.is_file()
    ]
    before = {str(p): digest(p) for p in paths}
    result = run.run_study(project)
    assert result["new_fits"] == 12 and result["reused_prior_controls"] == 6
    assert result["control_design_parity"] is True
    assert len(result["metrics"]) == 18 and len(run.figures(result)) == 8
    assert result["kaggle_score"] is None and not result["automatic_gpu_authorization"]
    assert all(digest(Path(p)) == h for p, h in before.items())


def test_interruption_reuses_three_fits_and_finishes_nine(project):
    with pytest.raises(TimeoutError, match="authored interruption"):
        run.run_study(project, max_new_fits=3)
    result = run.run_study(project)
    assert result["new_fits"] == 9 and result["reused_new_fits"] == 3


def test_complete_replay_never_fits_or_builds_features(project, monkeypatch):
    first = run.run_study(project)

    def forbidden(*args, **kwargs):
        pytest.fail("completed scientific work recomputed")

    monkeypatch.setattr(run, "fit_readout", forbidden)
    monkeypatch.setattr(run, "feature_bank", forbidden)
    assert run.run_study(project) == first
    record = json.loads((project / run.PRIVATE / "last_invocation.json").read_text())
    assert record == {"new_fits": 0, "reused": 12}


@pytest.mark.parametrize("kind", ["prediction", "bank", "public", "missing_bank"])
def test_completed_corruption_or_absence_stops(project, kind, monkeypatch):
    result = run.run_study(project)
    private = project / run.PRIVATE / result["run_id"]
    paths = {
        "prediction": private / "fold_0" / run.NEW[0] / "predictions.npz",
        "bank": private / "fold_0/feature_bank" / (next(iter(run.BANK_WIDTHS)) + ".npz"),
        "public": project / run.PUBLIC / "results.json",
        "missing_bank": private / "fold_0/feature_bank/complete.json",
    }
    path = paths[kind]
    if kind == "missing_bank":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b"corrupt")
    monkeypatch.setattr(run, "fit_readout", lambda *a, **k: pytest.fail("refit called"))
    with pytest.raises(ValueError):
        run.run_study(project)


def test_changed_prior_source_stops_before_fit(project, monkeypatch):
    (project / "scripts/matched_support_features.py").write_text("# changed\n")
    monkeypatch.setattr(run, "fit_readout", lambda *a, **k: pytest.fail("fit called"))
    with pytest.raises(ValueError, match="missing/changed"):
        run.run_study(project)


def test_prior_result_checksum_checked(project, monkeypatch):
    path = project / previous.PUBLIC / "results.json"
    path.write_text(path.read_text() + " ")
    monkeypatch.setattr(run, "fit_readout", lambda *a, **k: pytest.fail("fit called"))
    with pytest.raises(ValueError, match="missing/changed"):
        run.run_study(project)


def test_no_fit_if_cached_control_prediction_does_not_reproduce(project, monkeypatch):
    def forbidden(*args, **kwargs):
        raise ValueError("control parity differs")

    monkeypatch.setattr(run, "prepare_controls", forbidden)
    monkeypatch.setattr(run, "fit_readout", lambda *a, **k: pytest.fail("fit called"))
    with pytest.raises(ValueError, match="parity"):
        run.run_study(project)


def test_query_targets_not_supplied_to_feature_bank(project, monkeypatch):
    original = run.feature_bank

    def checked(bundle, *args, **kwargs):
        assert set(bundle["query"].columns) == {"row_id", "body", "rule"}
        return original(bundle, *args, **kwargs)

    monkeypatch.setattr(run, "feature_bank", checked)
    run.run_study(project)


def test_export_contains_no_raw_rows_private_arrays_or_models(project, tmp_path):
    run.run_study(project)
    path = run.export(project, tmp_path / "result.zip")
    with zipfile.ZipFile(path) as z:
        assert not any(
            n.startswith("data/") or n.endswith((".npz", ".npy", ".joblib")) for n in z.namelist()
        )
        assert all(
            hashlib.sha256(z.read(n)).hexdigest() == h
            for n, h in json.loads(z.read("SHA256SUMS.json")).items()
        )


def test_registration_and_primary_cannot_switch_to_secondary():
    spec = json.loads((Path(__file__).parents[1] / run.CONFIG).read_text())
    assert spec["new_fits"] == 12 and spec["cached_control_readouts"] == 6
    assert spec["independent_of_other_new_round"] is True
    metrics = [
        {"fold": fold, "variant": name, "auc": 0.5} for fold in range(2) for name in run.VARIANTS
    ]
    secondary = next(n for n in run.NEW if n != spec["primary"])
    for row in metrics:
        if row["variant"] == secondary:
            row["auc"] = 0.99
    comparisons = [
        {"candidate": spec["primary"], "reference": ref, "delta_auc": 0, "simultaneous_low": -0.01}
        for ref in run.PRIMARY_REFS
    ]
    pooled = [{"variant": name, "ranked_pooled_auc": 0.5} for name in run.VARIANTS]
    assert run.decide(metrics, comparisons, pooled, spec)["decision"] == "DO_NOT_PROMOTE_PRIMARY"


def test_incorrect_config_cannot_silently_relax_gate():
    spec = json.loads((Path(__file__).parents[1] / run.CONFIG).read_text())
    spec["minimum_macro_delta"] = 0
    with pytest.raises(ValueError, match="registered"):
        run.validate_config(spec)


def test_other_round_result_not_required(project):
    assert not (project / OTHER_ROUND_PUBLIC).exists()
    result = run.run_study(project)
    assert result["new_fits"] == 12


def test_python_source_static_style_and_version_guards():
    root = Path(__file__).parents[1]
    for name in run.SOURCES[:3]:
        if not name.endswith(".py"):
            continue
        source = (root / name).read_text()
        ast.parse(source, feature_version=(3, 12))
        assert max(map(len, source.splitlines())) <= 100
        tokens = list(tokenize.generate_tokens(iter(source.splitlines(keepends=True)).__next__))
        for tok in tokens:
            if tok.type == tokenize.STRING and tok.start[0] == tok.end[0]:
                assert len(tok.string) <= 78
    # No imported fixture names shadow parameter names in these tests.
    module = ast.parse(Path(__file__).read_text())
    imported = {
        a.asname or a.name for n in module.body if isinstance(n, ast.ImportFrom) for a in n.names
    }
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            assert not imported.intersection(a.arg for a in node.args.args)
