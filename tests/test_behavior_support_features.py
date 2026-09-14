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
from scripts import behavior_support_features as feat
from scripts import run_behavior_support_features as run
from scripts import run_matched_support_features as previous

OTHER_ROUND_PUBLIC = "reports/conditioned_geometry_features"


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


def test_two_full_feature_families():
    x, y, text, q, qt = inputs()
    out, stats = cross(x, y, text, q, qt)
    assert len(feat.ROLE_COLUMNS) == len(feat.CONTEXT_COLUMNS) == 6
    assert len(feat.NAMES) == 48
    assert set(out) == set(run.BANK_WIDTHS)
    assert all(t.shape == (len(x), 48) and v.shape == (len(q), 48) for t, v in out.values())
    assert len(stats) == 48
    assert all(0 <= s["mean_minimum_coverage"] <= 1 for s in stats)


def test_authored_descriptors_distinguish_requests_directives_and_quote():
    out = feat.descriptor_matrix(
        [
            "Should I hire a lawyer?",
            "You should sue the lawyer.",
            'He wrote "You should sue the lawyer."',
            "You should not sue the lawyer.",
        ]
    )
    assert out[0, 0] and not out[0, 2]
    assert out[1, 2] and not out[1, 0]
    assert out[2, 6] and not out[2, 2]
    assert out[3, 7] and out[3, 2]
    # No assertion about the moderation label is made.


def test_authored_link_roles_are_different():
    out = feat.descriptor_matrix(
        [
            "Buy from my shop https://shop.invalid today.",
            "Read this reference https://source.invalid for information.",
        ]
    )
    assert out[0, 9] and out[0, 10]
    assert out[1, 11] and not out[1, 9]


def test_rare_stratum_backoff_remains_finite():
    x, y, _, q, _ = inputs()
    rd, qd = np.zeros((len(x), 12), dtype=bool), np.ones((len(q), 12), dtype=bool)
    banks, stats = feat.conditional_geometry(x, y, rd, q, qd)
    assert all(s["zero_stratum_fraction"] == 1 for s in stats)
    assert all(np.isfinite(a).all() for a in banks.values())
    assert np.allclose(banks["conditioned"][:, 3::4], 0.2)
    assert np.allclose(banks["conditioned"][:, :3], 0)


def test_constant_descriptor_creates_no_conditional_contrast():
    x, y, _, q, _ = inputs()
    rd, qd = np.zeros((len(x), 12)), np.zeros((len(q), 12))
    banks, _ = feat.conditional_geometry(x, y, rd, q, qd)
    assert np.allclose(banks["conditioned"][:, 0::4], 0)
    assert np.allclose(banks["conditioned"][:, 1::4], 0)
    assert np.allclose(banks["conditioned"][:, 2::4], 0)
    assert np.allclose(banks["conditioned"][:, 3::4], 1)


def test_descriptor_only_control_contains_no_reference_label_information():
    x, y, text, q, qt = inputs()
    rd, qd = feat.descriptor_matrix(text), feat.descriptor_matrix(qt)
    a, _ = feat.conditional_geometry(x, y, rd, q, qd)
    b, _ = feat.conditional_geometry(-x, 1 - y, 1 - rd, q, qd)
    np.testing.assert_array_equal(a["descriptor_only"], b["descriptor_only"])
    np.testing.assert_array_equal(a["descriptor_only"][:, 0::4], qd)
    assert np.count_nonzero(a["descriptor_only"][:, 1::4]) == 0


def test_uniform_control_has_no_semantic_value_dependency():
    x, y, text, q, qt = inputs()
    rd, qd = feat.descriptor_matrix(text), feat.descriptor_matrix(qt)
    a, _ = feat.conditional_geometry(x, y, rd, q, qd)
    b, _ = feat.conditional_geometry(-x, y, rd, q, qd)
    np.testing.assert_array_equal(a["uniform"], b["uniform"])


def test_reference_descriptor_permutation_is_deterministic():
    x, y, text, q, qt = inputs()
    rd, qd = feat.descriptor_matrix(text), feat.descriptor_matrix(qt)
    a, _ = feat.conditional_geometry(x, y, rd, q, qd, seed=16)
    b, _ = feat.conditional_geometry(x, y, rd, q, qd, seed=16)
    np.testing.assert_array_equal(a["permuted"], b["permuted"])
    assert not np.allclose(a["conditioned"], a["permuted"])


def test_flip_reference_labels_reverses_signed_feature_contrasts():
    x, y, text, q, qt = inputs()
    rd, qd = feat.descriptor_matrix(text), feat.descriptor_matrix(qt)
    a, _ = feat.conditional_geometry(x, y, rd, q, qd)
    b, _ = feat.conditional_geometry(x, 1 - y, rd, q, qd)
    for offset in (0, 1, 2):
        np.testing.assert_allclose(
            a["conditioned"][:, offset::4], -b["conditioned"][:, offset::4], atol=1e-12
        )
    np.testing.assert_allclose(a["conditioned"][:, 3::4], b["conditioned"][:, 3::4])


@pytest.mark.parametrize(
    "bad", [{"temperature": 0}, {"temperature": float("nan")}, {"backoff": 0}, {"backoff": 1}]
)
def test_invalid_conditional_parameters_rejected(bad):
    x, y, text, q, qt = inputs()
    with pytest.raises(ValueError):
        feat.conditional_geometry(
            x, y, feat.descriptor_matrix(text), q, feat.descriptor_matrix(qt), **bad
        )


def test_descriptor_schema_and_nonbinary_fail():
    x, y, _, q, _ = inputs()
    with pytest.raises(ValueError, match="dimensions"):
        feat.conditional_geometry(x, y, np.zeros((len(x), 11)), q, np.zeros((len(q), 12)))
    with pytest.raises(ValueError, match="binary"):
        feat.conditional_geometry(x, y, np.full((len(x), 12), 0.5), q, np.zeros((len(q), 12)))


def test_descriptor_calls_never_request_query_targets():
    import inspect

    assert list(inspect.signature(feat.descriptor_matrix).parameters) == ["bodies"]


def test_observable_text_validation():
    for bad in ([], [None], [""], [1]):
        with pytest.raises(ValueError):
            feat.descriptor_matrix(bad)


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
