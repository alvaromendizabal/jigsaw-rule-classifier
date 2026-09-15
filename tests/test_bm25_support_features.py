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
from scripts import bm25_support_features as feat
from scripts import run_bm25_support_features as run


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
    from scripts import run_passage_support_features as r13
    from scripts import run_reference_consistency_features as r12
    from tests.test_passage_support_features import make_project as make13

    root = make13(root)
    thirteen = r13.run_study(root)
    source = Path(__file__).parents[1]
    for name in (*r12.SOURCES, r12.NOTEBOOK, *run.SOURCES, run.NOTEBOOK):
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            shutil.copyfile(source / name, p)
    prior = json.loads((root / r13.CONFIG).read_text())
    s12 = json.loads((root / r12.CONFIG).read_text())
    for n in (8, 9, 10, 11):
        for suffix in ("run_id", "results_sha256"):
            s12[f"round{n}_{suffix}"] = prior[f"round{n}_{suffix}"]
    s12["bootstrap_replicates"] = 40
    atomic_json(root / r12.CONFIG, s12)
    twelve = r12.run_study(root)
    spec = json.loads((root / run.CONFIG).read_text())
    for n in (8, 9, 10, 11):
        for suffix in ("run_id", "results_sha256"):
            spec[f"round{n}_{suffix}"] = prior[f"round{n}_{suffix}"]
    spec.update(
        round12_run_id=twelve["run_id"],
        round13_run_id=thirteen["run_id"],
        round12_results_sha256=digest(root / r12.PUBLIC / "results.json"),
        round13_results_sha256=digest(root / r13.PUBLIC / "results.json"),
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


@pytest.mark.parametrize("number", [8, 9, 10, 11, 12, 13])
def test_old_report_corruption_stops(project, monkeypatch, number):
    slug = {
        8: "conditioned_geometry_features",
        9: "behavior_support_features",
        10: "crossmodal_support_features",
        11: "multiprototype_features",
        12: "reference_consistency_features",
        13: "passage_support_features",
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


def test_bm25_has_expected_term_frequency_saturation():
    from scipy.sparse import csr_matrix

    ref = csr_matrix([[1.0], [10.0], [100.0]])
    score = feat.bm25_similarity(ref, csr_matrix([[1.0]]), np.ones(3), b=0)[0]
    assert 0 < score[0] < score[1] < score[2] < 1
    assert score[2] - score[1] < score[1] - score[0]


def test_length_penalty_changes_equal_term_counts():
    from scipy.sparse import csr_matrix

    ref = csr_matrix([[1.0], [1.0]])
    score = feat.bm25_similarity(ref, csr_matrix([[1.0]]), [1, 100])[0]
    assert score[0] > score[1]
    bzero = feat.bm25_similarity(ref, csr_matrix([[1.0]]), [1, 100], b=0)
    assert bzero[0, 0] == bzero[0, 1]


def test_binary_query_repetition_invariant():
    from scipy.sparse import csr_matrix

    ref = csr_matrix([[1.0, 2.0], [2.0, 1.0]])
    a = feat.bm25_similarity(ref, csr_matrix([[1.0, 1.0]]), [3, 3])
    b = feat.bm25_similarity(ref, csr_matrix([[5.0, 99.0]]), [3, 3])
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("lengths", [[1, np.nan], [1, -2], [1, np.inf], []])
def test_bad_reference_lengths(lengths):
    from scipy.sparse import csr_matrix

    with pytest.raises(ValueError):
        feat.bm25_similarity(csr_matrix([[1.0], [2.0]]), csr_matrix([[1.0]]), lengths)


def test_missing_terms_give_zero_not_nan():
    from scipy.sparse import csr_matrix

    score = feat.bm25_similarity(csr_matrix([[1.0, 0.0], [0.0, 1.0]]), csr_matrix((1, 2)), [1, 1])
    np.testing.assert_array_equal(score, np.zeros((1, 2)))


def test_shuffled_class_control_preserves_pooled_similarity():
    x, y, t, q, qt = inputs()
    banks, _ = feat.feature_block(x, y, t, q, qt, seed=9)
    assert banks["bm25"].shape == banks["label_null"].shape
    assert not np.array_equal(banks["bm25"], banks["label_null"])


def test_query_words_never_enter_vocabulary():
    x, y, t, q, qt = inputs()
    _, a = feat.feature_block(x, y, t, q, qt)
    _, b = feat.feature_block(x, y, t, q, ["zzqqrrss unseenquerytoken"] * len(q))
    assert [v["vocabulary_columns"] for v in a] == [v["vocabulary_columns"] for v in b]


def test_matrices_are_not_mutated():
    from scipy.sparse import csr_matrix

    ref, query = csr_matrix([[1.0, 0.0], [2.0, 1.0]]), csr_matrix([[2.0, 0.0]])
    a, b = ref.copy(), query.copy()
    feat.bm25_similarity(ref, query, [1, 3])
    np.testing.assert_array_equal(ref.toarray(), a.toarray())
    np.testing.assert_array_equal(query.toarray(), b.toarray())
