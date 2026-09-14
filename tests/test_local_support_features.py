from __future__ import annotations

import ast
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from jigsaw_rules.data import EXAMPLES, normalize
from jigsaw_rules.runtime import atomic_json, digest, environment
from scripts import local_support_features as feat
from scripts import run_local_support_features as run
from scripts.run_behavioral_features import protocol


def arrays():
    rng = np.random.default_rng(31)
    y = np.arange(48) % 2
    x = rng.normal(size=(48, 16))
    x[:, 0] += 2 * y - 1
    q = rng.normal(size=(9, 16))
    return x, y, q


def test_feature_schema_finite_and_immutable():
    x, y, q = arrays()
    before = x.copy(), y.copy(), q.copy()
    result = feat.geometry(x, y, q)
    assert result.shape == (9, 21)
    assert len(feat.BASIC) == 9 and len(feat.LOCAL) == 12
    assert np.isfinite(result).all()
    for old, new in zip(before, (x, y, q), strict=True):
        np.testing.assert_array_equal(old, new)


def test_query_batch_invariance_and_order():
    x, y, q = arrays()
    all_rows = feat.geometry(x, y, q)
    one = np.vstack([feat.geometry(x, y, row[None]) for row in q])
    np.testing.assert_allclose(all_rows, one, atol=1e-12)
    np.testing.assert_allclose(feat.geometry(x, y, q[::-1]), all_rows[::-1], atol=1e-12)


def test_positive_scaling_does_not_change_features():
    x, y, q = arrays()
    base = feat.geometry(x, y, q)
    np.testing.assert_allclose(feat.geometry(x * 6, y, q * 0.2), base, atol=1e-12)


def test_class_reversal_flips_label_evidence_not_radius():
    x, y, q = arrays()
    a, b = feat.geometry(x, y, q), feat.geometry(x, 1 - y, q)
    np.testing.assert_allclose(a[:, 6:9], -b[:, 6:9], atol=1e-12)
    np.testing.assert_allclose(a[:, 11], -b[:, 11], atol=1e-12)
    np.testing.assert_allclose(a[:, -1], b[:, -1], atol=1e-12)
    np.testing.assert_allclose(a[:, 12] + b[:, 12], 1, atol=1e-12)


def test_raw_geometry_matches_historical_primitives():
    from scripts.support_adaptation import prototype_features

    x, y, q = arrays()
    expected, _ = prototype_features(feat.unit(q), feat.unit(x), y)
    np.testing.assert_allclose(feat.geometry(x, y, q)[:, :9], expected, atol=1e-12)


def test_effective_support_fraction_valid():
    x, y, q = arrays()
    values = feat.geometry(x, y, q)
    assert np.all(values[:, 17:19] <= 1e-12)
    assert np.all(values[:, 17:19] >= -np.log(24) - 1e-12)
    assert np.all((values[:, 13] >= 0) & (values[:, 13] <= np.log(2) + 1e-12))


@pytest.mark.parametrize("bad", [0, -2, 26, 1.5])
def test_bad_neighbor_budget_stops(bad):
    x, y, q = arrays()
    with pytest.raises(ValueError, match="neighbors"):
        feat.geometry(x, y, q, neighbors=bad)


@pytest.mark.parametrize("issue", ["zero", "nan", "infinite", "wrong_dim"])
def test_vector_failures_stop(issue):
    x, y, q = arrays()
    if issue == "zero":
        x[0] = 0
    elif issue == "nan":
        x[0, 0] = np.nan
    elif issue == "infinite":
        q[0, 0] = np.inf
    else:
        q = q[:, :-1]
    with pytest.raises(ValueError):
        feat.geometry(x, y, q)


@pytest.mark.parametrize("kind", ["missing_class", "fractional", "misaligned"])
def test_invalid_labels_stop(kind):
    x, y, q = arrays()
    if kind == "missing_class":
        y[:] = 0
    elif kind == "fractional":
        y = y.astype(float)
        y[0] = 0.4
    else:
        y = y[:-1]
    with pytest.raises(ValueError):
        feat.geometry(x, y, q)


def test_seeded_shuffle_is_deterministic_and_changes_features():
    x, y, q = arrays()
    a = feat.geometry(x, y, q, shuffle_seed=20)
    np.testing.assert_array_equal(a, feat.geometry(x, y, q, shuffle_seed=20))
    assert not np.allclose(a, feat.geometry(x, y, q))


def test_crossfit_excludes_duplicate_text_and_own_labels():
    x, y, q = arrays()
    bodies = [f"group {i // 2}" for i in range(len(x))]
    a, v, meta = feat.cross_fitted(x, y, bodies, q)
    groups = np.array([normalize(t) for t in bodies])
    _, held = next(GroupKFold(n_splits=3).split(x, groups=groups))
    changed = y.copy()
    changed[held] = 1 - changed[held]
    b, _, _ = feat.cross_fitted(x, changed, bodies, q)
    np.testing.assert_allclose(a[held], b[held], atol=1e-12)
    assert a.shape == (48, 21) and v.shape == (9, 21)
    assert all(m["self_text_overlap"] == 0 for m in meta)


def test_crossfit_reference_statistics_do_not_depend_on_query():
    x, y, q = arrays()
    bodies = [f"item {i}" for i in range(len(x))]
    a, _, _ = feat.cross_fitted(x, y, bodies, q)
    b, _, _ = feat.cross_fitted(x, y, bodies, q * -500)
    np.testing.assert_array_equal(a, b)


def test_duplicate_vectors_have_finite_radii_floor():
    x = np.ones((12, 6))
    result = feat.geometry(x, np.arange(12) % 2, x[:3])
    assert np.isfinite(result).all()


def make_project(root):
    source = Path(__file__).parents[1]
    config = json.loads((source / run.CONFIG).read_text())
    names = (
        set(run.SOURCES)
        | set(config["builder_sources"])
        | {
            run.NOTEBOOK,
            "scripts/feature_value_audit.py",
        }
    )
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, path)
    rows = []
    for rule_index, rule in enumerate(("No Advertising", "No legal advice")):
        for i in range(24):
            row = {
                "row_id": rule_index * 100 + i,
                "body": f"query {rule_index} {i}",
                "rule": rule,
                "subreddit": "authored",
                "rule_violation": i % 2,
            }
            for c in EXAMPLES:
                row[c] = f"example {rule_index} {i} {c}"
            rows.append(row)
    frame = pd.DataFrame(rows)
    (root / "data/raw").mkdir(parents=True)
    frame.to_csv(root / "data/raw/train.csv", index=False)
    plan, cohorts = protocol(frame, [24, 24])
    rng = np.random.default_rng(89)
    pins, scores, counts = [], [], []
    for fold, item in enumerate(plan["folds"]):
        train = pd.DataFrame(item["training"])
        query = pd.DataFrame(item["queries"])
        yv = frame.set_index("row_id").loc[query.row_id].rule_violation.to_numpy()
        yt = train.rule_violation.to_numpy()
        data = {"query_row_ids": query.row_id.to_numpy()}
        for model in ("frozen", "adapted"):
            for cohort, labels in (("train", yt), ("query", yv)):
                vectors = rng.normal(size=(len(labels), 16))
                vectors[:, 0] += 2 * labels - 1
                margin = 1.2 * (2 * labels - 1) + rng.normal(size=len(labels))
                data[f"{cohort}_{model}_vectors"] = feat.unit(vectors)
                data[f"{cohort}_{model}_scores"] = np.column_stack(
                    [
                        expit(margin),
                        margin,
                        np.full(len(labels), 0.9),
                    ]
                )
        target = root / "runs/feature_value_audit/reference" / f"fold_{fold}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **data)
        pins.append(
            {
                "name": target.name,
                "bytes": target.stat().st_size,
                "sha256": digest(target),
                "key": "authored-not-remote",
            }
        )
        scores.append(
            {
                "fold": fold,
                "model": "qwen_adapted_reference",
                "auc": float(roc_auc_score(yv, data["query_adapted_scores"][:, 1])),
            }
        )
        counts.append(int(train.rule.eq(item["rule"]).sum()))
    prior_ident = {
        "source": {
            "scripts/feature_value_audit.py": digest(root / "scripts/feature_value_audit.py")
        },
        "environment": environment(),
        "config": {"reference_files": pins},
    }
    prior_id = run.json_hash(prior_ident)[:20]
    prior = {
        "run_id": prior_id,
        "identity": prior_ident,
        "metrics": scores,
        "cohorts": cohorts,
        "authored_synthetic_data": True,
    }
    audit = root / "reports/feature_value_audit/audit.json"
    atomic_json(audit, prior)
    atomic_json(
        root / "runs/feature_value_audit" / prior_id / "complete.json",
        {
            "identity": prior_ident,
            "artifacts": {"audit.json": digest(audit)},
        },
    )
    config.update(
        audit_run_id=prior_id,
        audit_sha256=digest(audit),
        train_sha256=digest(root / "data/raw/train.csv"),
        query_counts=[24, 24],
        expected_same_rule_training=counts,
        dimensions=16,
        bootstrap_replicates=40,
        reference_files=pins,
    )
    atomic_json(root / run.CONFIG, config)
    return root


@pytest.fixture
def project(tmp_path):
    return make_project(tmp_path / "project")


def test_input_preflight_has_no_query_targets_in_fitting_bundle(project):
    config = json.loads((project / run.CONFIG).read_text())
    bundles, _, _, _ = run.load_inputs(project, config)
    for b in bundles:
        assert set(b["query"].columns) == {"row_id", "body", "rule"}
        assert b["train"].rule.nunique() == 1
        assert set(b["train"].body.map(normalize)).isdisjoint(b["query"].body.map(normalize))


def test_full_study_and_no_prior_modifications(project):
    paths = [
        p
        for directory in ("runs/feature_value_audit", "reports/feature_value_audit", "data")
        for p in (project / directory).rglob("*")
        if p.is_file()
    ]
    before = {str(p): digest(p) for p in paths}
    result = run.run_study(project)
    assert result["new_fits"] == 12 and result["cached_qwen_controls"] == 2
    assert len(result["metrics"]) == 14 and len(result["comparisons"]) == 11
    assert result["kaggle_score"] is None and not result["automatic_gpu_authorization"]
    assert len(run.figures(result)) == 8
    assert all(digest(Path(p)) == h for p, h in before.items())


def test_interruption_reuses_three_completed_fits(project):
    with pytest.raises(TimeoutError, match="authored interruption"):
        run.run_study(project, max_new_fits=3)
    result = run.run_study(project)
    assert result["new_fits"] == 9 and result["reused_new_fits"] == 3


def test_completed_replay_never_fits_or_rebuilds_features(project, monkeypatch):
    first = run.run_study(project)

    def forbidden(*args, **kwargs):
        pytest.fail("completed work recomputed")

    monkeypatch.setattr(run, "fit_readout", forbidden)
    monkeypatch.setattr(run, "prepare_features", forbidden)
    assert run.run_study(project) == first
    assert json.loads((project / run.PRIVATE / "last_invocation.json").read_text())["new_fits"] == 0


@pytest.mark.parametrize("kind", ["prediction", "features", "public"])
def test_corruption_stops_instead_of_refitting(project, kind, monkeypatch):
    result = run.run_study(project)
    private = project / run.PRIVATE / result["run_id"]
    paths = {
        "prediction": private / "fold_0/frozen_all/predictions.npz",
        "features": private / "fold_0/features_frozen/features.npz",
        "public": project / run.PUBLIC / "results.json",
    }
    path = paths[kind]
    path.write_bytes(path.read_bytes() + b"CORRUPTION")
    monkeypatch.setattr(run, "fit_readout", lambda *a, **k: pytest.fail("refit"))
    with pytest.raises(ValueError):
        run.run_study(project)


@pytest.mark.parametrize(
    "path",
    [
        "data/raw/train.csv",
        "scripts/support_adaptation.py",
        "reports/feature_value_audit/audit.json",
    ],
)
def test_changed_input_stops_before_fit(project, path, monkeypatch):
    p = project / path
    p.write_bytes(p.read_bytes() + b"changed")
    monkeypatch.setattr(run, "fit_readout", lambda *a, **k: pytest.fail("fit"))
    with pytest.raises(ValueError):
        run.run_study(project)


def test_missing_reference_stops_without_download(project):
    (project / "runs/feature_value_audit/reference/fold_1.npz").unlink()
    with pytest.raises(ValueError, match="missing/changed input"):
        run.run_study(project)


def test_secondary_winner_not_promoted():
    spec = {"primary": "frozen_all", "minimum_macro_delta": 0.003}
    metrics = [
        {"fold": f, "variant": n, "auc": 0.99 if n == "adapted_all" else 0.5}
        for f in range(2)
        for n in (run.REFERENCE, *run.NEW)
    ]
    comparisons = [
        {"candidate": a, "reference": b, "delta_auc": 0, "simultaneous_low": -0.1}
        for a, b, _ in run.CONTRASTS
    ]
    pooled = [{"variant": n, "ranked_pooled_auc": 0.5} for n in (run.REFERENCE, *run.NEW)]
    assert run.decide(metrics, comparisons, pooled, spec)["decision"] == "DO_NOT_PROMOTE_PRIMARY"


def test_readout_scaler_does_not_use_evaluation_vectors():
    x, y, q = arrays()
    spec = {"C": 1, "solver": "liblinear", "max_iter": 2000, "seed": 2025}
    a = run.fit_readout(x, y, q, np.ones(len(y)), spec)
    b = run.fit_readout(x, y, q * 1000, np.ones(len(y)), spec)
    for name in ("coefficients", "keep", "mean", "scale", "intercept"):
        np.testing.assert_array_equal(a[name], b[name])


def test_private_data_not_in_export(project, tmp_path):
    run.run_study(project)
    path = run.export(project, tmp_path / "return.zip")
    with zipfile.ZipFile(path) as z:
        assert not any(n.startswith("data/") or n.endswith((".npz", ".npy")) for n in z.namelist())
        checks = json.loads(z.read("SHA256SUMS.json"))
        assert len(checks) == len(z.namelist()) - 1
        for n, h in checks.items():
            assert hashlib.sha256(z.read(n)).hexdigest() == h


def test_notebook_is_display_only_and_eight_cells():
    import nbformat

    book = nbformat.read(Path(__file__).parents[1] / run.NOTEBOOK, as_version=4)
    cells = [c for c in book.cells if c.cell_type == "code"]
    assert len(cells) == 8
    assert not any("run_study(" in c.source or "fit(" in c.source for c in cells)


def test_static_new_source_safety():
    # Prevent the long literals and fixture-alias errors encountered previously.
    import tokenize

    root = Path(__file__).parents[1]
    for name in run.SOURCES:
        if not name.endswith(".py"):
            continue
        text = (root / name).read_text()
        assert max(map(len, text.splitlines())) <= 100, name
        ast.parse(text, feature_version=(3, 12))
        with (root / name).open("rb") as stream:
            for t in tokenize.tokenize(stream.readline):
                if t.type == tokenize.STRING and t.start[0] == t.end[0]:
                    assert len(t.string) <= 85, (name, t.start)
