from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

from jigsaw_rules.data import normalize
from jigsaw_rules.runtime import atomic_json, digest
from scripts import reference_consistency_features as feat
from scripts import run_reference_consistency_features as run


def inputs(n=42, d=16):
    rng = np.random.default_rng(18)
    x, q = rng.normal(size=(n, d)), rng.normal(size=(8, d))
    y = np.arange(n) % 2
    x[:, 0] += (2 * y - 1) * 0.4
    forms = [
        "Should I hire a lawyer?",
        "You should sue.",
        "Buy from my shop.",
        "He quoted the rule.",
        "General discussion of local events.",
    ]
    texts = np.asarray([forms[i % 5] + f" reference{i}" for i in range(n)])
    qt = [forms[i % 5] + f" query{i}" for i in range(8)]
    return x, y, texts, q, qt


def cross(x, y, text, q, qt):
    return feat.cross_fitted(x, y, text, ["same rule"] * len(x), q, qt, seed=7)


def test_query_values_cannot_change_training_features():
    x, y, text, q, qt = inputs()
    a, _ = cross(x, y, text, q, qt)
    b, _ = cross(x, y, text, -q * 22, ["changed query " + str(i) for i in range(8)])
    for mode in a:
        np.testing.assert_array_equal(a[mode][0], b[mode][0])


def test_held_group_labels_cannot_enter_its_own_features():
    x, y, text, q, qt = inputs()
    a, _ = cross(x, y, text, q, qt)
    _, held = next(GroupKFold(3).split(x, groups=[normalize(t) for t in text]))
    changed = y.copy()
    changed[held] = 1 - changed[held]
    b, _ = cross(x, changed, text, q, qt)
    for mode in a:
        np.testing.assert_allclose(a[mode][0][held], b[mode][0][held], atol=1e-12)


def test_reference_order_invariant_including_nulls():
    x, y, text, q, qt = inputs()
    a, _ = cross(x, y, text, q, qt)
    b, _ = cross(x[::-1], y[::-1], text[::-1], q, qt)
    for mode in a:
        np.testing.assert_allclose(a[mode][0][::-1], b[mode][0], atol=1e-10)
        np.testing.assert_allclose(a[mode][1], b[mode][1], atol=1e-10)


def test_query_order_and_batch_invariance():
    x, y, text, q, qt = inputs()
    a, _ = cross(x, y, text, q, qt)
    b, _ = cross(x, y, text, q[::-1], qt[::-1])
    one, _ = cross(x, y, text, q[:1], qt[:1])
    for mode in a:
        np.testing.assert_allclose(a[mode][1][::-1], b[mode][1], atol=1e-10)
        np.testing.assert_allclose(a[mode][1][:1], one[mode][1], atol=1e-10)


def test_inputs_not_mutated():
    x, y, text, q, qt = inputs()
    saved = [v.copy() for v in (x, y, text, q)]
    cross(x, y, text, q, qt)
    for old, new in zip(saved, (x, y, text, q), strict=True):
        np.testing.assert_array_equal(old, new)


@pytest.mark.parametrize(
    "fault",
    [
        "zero",
        "nan",
        "infinity",
        "dimension",
        "label",
        "single_class",
        "duplicate",
        "overlap",
        "mixed_rules",
        "alignment",
        "empty_text",
        "missing_query",
        "empty_query",
        "few_rows",
    ],
)
def test_bad_inputs_rejected(fault):
    x, y, text, q, qt = inputs()
    rules = ["same rule"] * len(x)
    if fault == "zero":
        x[0] = 0
    elif fault == "nan":
        q[0, 0] = np.nan
    elif fault == "infinity":
        x[0, 0] = np.inf
    elif fault == "dimension":
        q = q[:, :-1]
    elif fault == "label":
        y[0] = 5
    elif fault == "single_class":
        y[:] = 0
    elif fault == "duplicate":
        text[1] = text[0]
    elif fault == "overlap":
        qt[0] = text[0]
    elif fault == "mixed_rules":
        rules[0] = "different rule"
    elif fault == "alignment":
        y = y[:-1]
    elif fault == "empty_text":
        text[0] = ""
    elif fault == "missing_query":
        qt = qt[:-1]
    elif fault == "empty_query":
        qt[0] = ""
    else:
        x, y, text, rules = x[:8], y[:8], text[:8], rules[:8]
    with pytest.raises(ValueError):
        feat.cross_fitted(x, y, text, rules, q, qt)


def test_dimensions_families_and_finite_outputs():
    x, y, text, q, qt = inputs()
    banks, stats = cross(x, y, text, q, qt)
    assert len(feat.NAMES) == 48
    assert len(set(feat.NAMES)) == 48
    assert set(banks) == set(run.BANK_WIDTHS)
    assert all(a.shape == (42, 48) and b.shape == (8, 48) for a, b in banks.values())
    assert all(np.isfinite(v).all() for pair in banks.values() for v in pair)
    assert all(s["self_overlap"] == 0 for s in stats)
    assert {s["inner_fold"] for s in stats} == {"0", "1", "2", "outer_query"}


def make_project(root):
    from scripts import run_crossmodal_support_features as r10
    from scripts import run_multiprototype_features as r11
    from tests.test_multiprototype_features import make_project as make_round11

    root = make_round11(root)
    eleven = r11.run_study(root)
    source = Path(__file__).parents[1]
    for name in (*r10.SOURCES, r10.NOTEBOOK):
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            shutil.copyfile(source / name, p)
    previous = json.loads((root / r11.CONFIG).read_text())
    s10 = json.loads((root / r10.CONFIG).read_text())
    for n in (8, 9):
        for suffix in ("run_id", "results_sha256"):
            s10[f"round{n}_{suffix}"] = previous[f"round{n}_{suffix}"]
    s10["bootstrap_replicates"] = 40
    atomic_json(root / r10.CONFIG, s10)
    ten = r10.run_study(root)
    for name in (*run.SOURCES, run.NOTEBOOK):
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            shutil.copyfile(source / name, p)
    spec = json.loads((root / run.CONFIG).read_text())
    for n in (8, 9):
        for suffix in ("run_id", "results_sha256"):
            spec[f"round{n}_{suffix}"] = previous[f"round{n}_{suffix}"]
    spec.update(
        round10_run_id=ten["run_id"],
        round11_run_id=eleven["run_id"],
        round10_results_sha256=digest(root / r10.PUBLIC / "results.json"),
        round11_results_sha256=digest(root / r11.PUBLIC / "results.json"),
        bootstrap_replicates=40,
    )
    atomic_json(root / run.CONFIG, spec)
    return root


@pytest.fixture(scope="module")
def study_template(tmp_path_factory):
    return make_project(tmp_path_factory.mktemp("new-round") / "project")


@pytest.fixture
def project(study_template, tmp_path):
    return Path(shutil.copytree(study_template, tmp_path / "project"))


def test_full_study_and_preservation(project):
    old = [
        p
        for group in (
            "data",
            "runs/behavior_support_features",
            "reports/behavior_support_features",
            "runs/conditioned_geometry_features",
            "reports/conditioned_geometry_features",
        )
        for p in (project / group).rglob("*")
        if p.is_file()
    ]
    before = {p: digest(p) for p in old}
    result = run.run_study(project)
    assert result["new_fits"] == 12 and result["reused_prior_controls"] == 10
    assert len(result["metrics"]) == 22 and len(run.figures(result)) == 8
    assert result["control_design_parity"] and result["kaggle_score"] is None
    assert not result["automatic_gpu_authorization"]
    assert all(digest(p) == h for p, h in before.items())


def test_interrupt_three_and_resume_nine(project):
    with pytest.raises(TimeoutError, match="authored interruption"):
        run.run_study(project, max_new_fits=3)
    result = run.run_study(project)
    assert result["new_fits"] == 9 and result["reused_new_fits"] == 3


def test_replay_never_calls_fit_or_features(project, monkeypatch):
    first = run.run_study(project)

    def forbidden(*args, **kwargs):
        pytest.fail("unexpected scientific recomputation")

    monkeypatch.setattr(run, "fit_readout", forbidden)
    monkeypatch.setattr(run, "feature_bank", forbidden)
    assert run.run_study(project) == first
    assert json.loads((project / run.PRIVATE / "last_invocation.json").read_text()) == {
        "new_fits": 0,
        "reused": 12,
    }


@pytest.mark.parametrize("kind", ["prediction", "bank", "missing_bank", "public"])
def test_completed_corruption_stops(project, monkeypatch, kind):
    result = run.run_study(project)
    folder = project / run.PRIVATE / result["run_id"] / "fold_0"
    target = {
        "prediction": folder / run.NEW[0] / "predictions.npz",
        "bank": folder / "feature_bank" / (next(iter(run.BANK_WIDTHS)) + ".npz"),
        "missing_bank": folder / "feature_bank/complete.json",
        "public": project / run.PUBLIC / "results.json",
    }[kind]
    if kind == "missing_bank":
        target.unlink()
    else:
        target.write_bytes(target.read_bytes() + b"corrupt")
    monkeypatch.setattr(run, "fit_readout", lambda *a, **k: pytest.fail("fit called"))
    with pytest.raises(ValueError):
        run.run_study(project)


@pytest.mark.parametrize("number", [8, 9, 10, 11])
def test_old_report_corruption_stops(project, monkeypatch, number):
    slug = {
        8: "conditioned_geometry_features",
        9: "behavior_support_features",
        10: "crossmodal_support_features",
        11: "multiprototype_features",
    }[number]
    p = project / "reports" / slug / "results.json"
    p.write_bytes(p.read_bytes() + b" ")
    monkeypatch.setattr(run, "fit_readout", lambda *a, **k: pytest.fail("fit called"))
    with pytest.raises(ValueError):
        run.run_study(project)


def test_saved_anchor_must_reproduce_before_fit(project, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("anchor parity failed")

    monkeypatch.setattr(run, "prepare_controls", fail)
    monkeypatch.setattr(run, "fit_readout", lambda *a, **k: pytest.fail("fit called"))
    with pytest.raises(ValueError, match="parity"):
        run.run_study(project)


def test_feature_bank_never_receives_query_targets(project, monkeypatch):
    original = run.feature_bank

    def checked(bundle, *args, **kwargs):
        assert set(bundle["query"]) == {"row_id", "body", "rule"}
        return original(bundle, *args, **kwargs)

    monkeypatch.setattr(run, "feature_bank", checked)
    run.run_study(project)


def test_export_allowlist_no_private_data(project, tmp_path):
    run.run_study(project)
    output = run.export(project, tmp_path / "return.zip")
    with zipfile.ZipFile(output) as archive:
        assert not any(
            n.startswith("data/") or n.endswith((".npz", ".npy", ".joblib"))
            for n in archive.namelist()
        )
        for name, checksum in json.loads(archive.read("SHA256SUMS.json")).items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == checksum


def test_unchanged_registration():
    spec = json.loads((Path(__file__).parents[1] / run.CONFIG).read_text())
    run.validate_config(spec)
    assert spec["new_fits"] == 2 * len(run.NEW) == 12
    assert spec["cached_control_readouts"] == 10
    assert spec["new_primary_columns"] == 48
    changed = {**spec, "primary": run.NEW[0]}
    with pytest.raises(ValueError):
        run.validate_config(changed)


def test_no_silent_promotion_of_secondary():
    spec = json.loads((Path(__file__).parents[1] / run.CONFIG).read_text())
    metrics = [
        {"variant": name, "fold": fold, "auc": 0.99 if name == run.NEW[0] else 0.5}
        for name in run.VARIANTS
        for fold in range(2)
    ]
    contrasts = [
        {"candidate": spec["primary"], "reference": name, "delta_auc": 0, "simultaneous_low": -0.01}
        for name in run.PRIMARY_REFS
    ]
    pooled = [{"variant": name, "ranked_pooled_auc": 0.5} for name in run.VARIANTS]
    assert run.decide(metrics, contrasts, pooled, spec)["decision"] == "DO_NOT_PROMOTE_PRIMARY"


def test_promotion_requires_every_comparator_and_policy():
    spec = json.loads((Path(__file__).parents[1] / run.CONFIG).read_text())
    metrics = [
        {"variant": n, "fold": f, "auc": 0.6 if n == spec["primary"] else 0.5}
        for n in run.VARIANTS
        for f in range(2)
    ]
    comparisons = [
        {"candidate": spec["primary"], "reference": n, "delta_auc": 0.1, "simultaneous_low": 0.01}
        for n in run.PRIMARY_REFS
    ]
    pooled = [
        {"variant": n, "ranked_pooled_auc": 0.6 if n == spec["primary"] else 0.5}
        for n in run.VARIANTS
    ]
    assert run.decide(metrics, comparisons, pooled, spec)["decision"].startswith("ELIGIBLE")
    comparisons[-1]["simultaneous_low"] = -0.001
    assert run.decide(metrics, comparisons, pooled, spec)["decision"] == "DO_NOT_PROMOTE_PRIMARY"
    comparisons[-1]["simultaneous_low"] = 0.01
    next(m for m in metrics if m["variant"] == spec["primary"] and m["fold"] == 1)["auc"] = 0.4
    assert run.decide(metrics, comparisons, pooled, spec)["decision"] == "DO_NOT_PROMOTE_PRIMARY"


def test_prior_checks_fail_without_changing_existing_artifacts(project, monkeypatch):
    before = {p: digest(p) for p in project.rglob("*") if p.is_file()}

    def fail(*args, **kwargs):
        raise ValueError("prior blocked")

    monkeypatch.setattr(run, "verify_prior", fail)
    with pytest.raises(ValueError, match="prior blocked"):
        run.run_study(project)
    assert all(digest(p) == h for p, h in before.items())


def test_source_line_lengths():
    root = Path(__file__).parents[1]
    for name in run.SOURCES[:3]:
        assert all(len(line) <= 100 for line in (root / name).read_text().splitlines())


def test_reference_self_exclusion_and_bounds():
    x, y, _, _, _ = inputs()
    quality, centrality, diag = feat.reference_quality(x, y)
    assert quality.shape == (42, 2) and centrality.shape == (42,)
    assert ((quality >= 0.25) & (quality <= 1)).all()
    assert diag["self_neighbors"] == 0 and diag["neighbors_per_reference"] == 7


def test_geometric_control_independent_of_labels():
    x, y, _, _, _ = inputs()
    _, a, _ = feat.reference_quality(x, y)
    _, b, _ = feat.reference_quality(x, np.random.default_rng(99).permutation(y))
    np.testing.assert_array_equal(a, b)


def test_class_flip_swaps_observed_features():
    x, y, text, q, qt = inputs()
    a, _ = feat.feature_block(x, y, text, q, qt)
    b, _ = feat.feature_block(x, 1 - y, text, q, qt)
    for offset in (0, 24):
        np.testing.assert_allclose(
            a["observed"][:, offset : offset + 8], b["observed"][:, offset + 8 : offset + 16]
        )
        np.testing.assert_allclose(
            a["observed"][:, offset + 16 : offset + 24],
            -b["observed"][:, offset + 16 : offset + 24],
        )


def test_null_preserves_within_class_quality_distribution():
    x, y, text, q, qt = inputs()
    _, stats = feat.feature_block(x, y, text, q, qt)
    for cls in (0, 1):
        a = next(v for v in stats if v["class"] == cls and v["mode"] == "observed")
        b = next(v for v in stats if v["class"] == cls and v["mode"] == "quality_null")
        for key in ("mean_quality", "minimum_quality", "maximum_quality"):
            assert a[key] == pytest.approx(b[key])


def test_uniform_quality_has_zero_reweighting_effect():
    sim = np.random.default_rng(1).uniform(-1, 1, (5, 9))
    out = feat.class_statistics(sim, np.ones(9) * 0.7)
    np.testing.assert_allclose(out[:, [0, 1, 3, 5, 7]], 0, atol=1e-12)
    np.testing.assert_allclose(out[:, 2], 0.7)


def test_rotation_invariance_of_consistency():
    x, y, text, q, qt = inputs()
    rot, _ = np.linalg.qr(np.random.default_rng(77).normal(size=(16, 16)))
    a, _ = feat.feature_block(x, y, text, q, qt)
    b, _ = feat.feature_block(3 * x @ rot, y, text, 5 * q @ rot, qt)
    for name in a:
        np.testing.assert_allclose(a[name], b[name], atol=1e-10)


def test_rare_examples_are_never_removed_or_relabelled():
    x, y, _, _, _ = inputs()
    before = y.copy()
    quality, _, _ = feat.reference_quality(x, y)
    assert len(quality) == len(x) and (quality > 0).all()
    np.testing.assert_array_equal(before, y)


@pytest.mark.parametrize("bad", [np.nan, 0.1, 1.01])
def test_invalid_quality_rejected(bad):
    with pytest.raises(ValueError):
        feat.class_statistics(np.ones((2, 4)), np.array([bad, 0.5, 0.5, 0.5]))


def test_no_query_target_argument_or_score_dependency():
    import inspect

    assert "query_labels" not in inspect.signature(feat.cross_fitted).parameters
    assert "scores" not in inspect.signature(feat.cross_fitted).parameters
