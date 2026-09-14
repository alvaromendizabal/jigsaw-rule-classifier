from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts import run_scope_lexical_features as run
from scripts import scope_lexical_features as sf
from scripts.evidence_features import RuleEvidence
from tests.test_evidence_features import policy_template  # noqa: F401
from tests.test_evidence_features import study_template as evidence_template  # noqa: F401


def labels(text):
    _, tokens, attribution, negation = sf.token_scopes(text)
    triples = zip(tokens, attribution, negation, strict=True)
    return [(t.group(), int(a), int(n)) for t, a, n in triples]


def test_quoted_and_authored_same_word_separated():
    rows = labels('"sue" then sue')
    assert rows[0] == ("sue", 1, 0)
    assert rows[-1] == ("sue", 0, 0)


@pytest.mark.parametrize("text", ['`"sue"`', '```\n"sue"\n```'])
def test_code_takes_precedence_over_quotes(text):
    assert next(r[1] for r in labels(text) if r[0] == "sue") == 2


def test_curly_quotes_and_block_quotes_recognized():
    assert labels("“sue”") == [("sue", 1, 0)]
    assert labels("> sue\norder")[0] == ("sue", 1, 0)
    assert labels("> sue\norder")[-1] == ("order", 0, 0)


def test_single_quote_prose_is_an_explicit_limitation():
    assert labels("'sue'") == [("sue", 0, 0)]


@pytest.mark.parametrize("text", ["do not sue", "never sue", "don't sue", "can’t sue"])
def test_negation_window_flags_action_but_assigns_no_class(text):
    assert next(r[2] for r in labels(text) if r[0] == "sue") == 1


@pytest.mark.parametrize("text", ["not only sue", "not just sue", "no wonder sue"])
def test_pseudo_negators_excluded(text):
    assert next(r[2] for r in labels(text) if r[0] == "sue") == 0


@pytest.mark.parametrize("boundary", [".", ";", "?", "!", "\n", " but ", " however "])
def test_negation_stops_at_declared_boundary(boundary):
    assert labels("not cancel" + boundary + "sue")[-1][2] == 0


def test_negation_window_does_not_cross_quotation_region():
    assert labels('not "cancel" sue')[-1][2] == 0
    assert labels('"not sue" buy')[-1][2] == 0
    assert labels("not `cancel` sue")[-1][2] == 0


def test_four_token_window_is_exact():
    rows = labels("not alpha bravo charlie delta echo")
    assert [r[2] for r in rows] == [0, 1, 1, 1, 1, 0]


@pytest.mark.parametrize("text", [None, "", " ", 2])
def test_invalid_text_stops(text):
    with pytest.raises(ValueError, match="nonempty"):
        sf.token_scopes(text)


@pytest.mark.parametrize("window", [True, 0, 8, 4.0])
def test_fixed_window_not_silently_tuned(window):
    with pytest.raises(ValueError, match="four"):
        sf.token_scopes("not sue", window=window)


def test_occurrence_counts_exactly_match_existing_word_analyzer():
    texts = [
        'I should "not sue"; buy now.',
        "not only good — `real code` \n> quoted words",
        "cannot can’t don't café Unicode 中文 text",
        "lawyer lawyer lawyer",
    ]
    vec = TfidfVectorizer(ngram_range=(1, 2)).fit(texts)
    full, blocks, _ = sf.occurrence_counts(texts, vec.vocabulary_)
    for i, text in enumerate(texts):
        counts = Counter(vec.build_analyzer()(text))
        expected = np.zeros(len(vec.vocabulary_))
        for term, count in counts.items():
            expected[vec.vocabulary_[term]] = count
        np.testing.assert_array_equal(full[i].toarray().ravel(), expected)
    d = full.shape[1]
    for family, matrix in blocks.items():
        recovered = sum(matrix[:, k * d : (k + 1) * d] for k in range(len(sf.CHANNELS[family])))
        assert (recovered != full).nnz == 0


def test_deleting_a_quote_does_not_invent_a_bigram():
    terms = {"sue": 0, "buy": 1, "order": 2, "sue order": 3}
    full, _, _ = sf.occurrence_counts(['sue "buy" order'], terms)
    assert full[0, 3] == 0


def test_energy_controls_keep_each_original_term_squared_magnitude():
    value = sparse.csr_matrix([[0.6, 0.8], [0, 0]])
    counts = sparse.csr_matrix([[4, 2], [0, 0]])
    channels = sparse.csr_matrix([[1, 0, 3, 2], [0, 0, 0, 0]])
    a, b, error = sf.energy_partition(value, counts, channels, 2)
    assert a.shape == b.shape == (2, 4)
    assert error < 1e-12
    energy = a[:, :2].power(2) + a[:, 2:].power(2)
    np.testing.assert_allclose(energy.toarray(), value.power(2).toarray())
    np.testing.assert_allclose(sf.row_norms(a), sf.row_norms(b))
    assert a[1].nnz == 0


def test_one_context_equals_collapsed_control():
    value = sparse.csr_matrix([[0.6, 0.8]])
    counts = sparse.csr_matrix([[4, 2]])
    channels = sparse.csr_matrix([[4, 2, 0, 0]])
    a, b, _ = sf.energy_partition(value, counts, channels, 2)
    np.testing.assert_array_equal(a.toarray(), b.toarray())


def test_invalid_count_partition_stops():
    with pytest.raises(ValueError, match="partition"):
        sf.energy_partition([[1]], [[2]], [[3, 0]], 2)
    with pytest.raises(ValueError, match="dimensions"):
        sf.energy_partition([[1]], [[2]], [[2]], 2)


def test_no_query_fit_targets_or_vocabulary_change():
    frame = pd.DataFrame(
        {"rule": ["a"] * 4, "body": ["sue now", '"sue" now', "buy now", "not buy"]}
    )
    spec = {"word_features": 100}
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100, sublinear_tf=True)
    x = v.fit_transform(("RULE: " + frame.rule + "\nBODY: " + frame.body).tolist())
    evidence = RuleEvidence().fit(x, [1, 1, 0, 0], frame.rule)
    fitted = sf.ScopedLexicon().fit(frame, spec, v.get_feature_names_out(), evidence)
    before = evidence.weights("a", "rule").copy()
    expected = fitted.transform(frame)[0]["attribution"]["scope"].copy()
    fitted.transform(pd.DataFrame({"body": ["unseen_token not hello"], "rule": ["new"]}))
    np.testing.assert_array_equal(before, evidence.weights("a", "rule"))
    assert (expected != fitted.transform(frame)[0]["attribution"]["scope"]).nnz == 0
    with pytest.raises(ValueError, match="body and rule only"):
        fitted.transform(frame.assign(rule_violation=1))


def test_declared_counts_primary_and_no_gpu():
    cfg = json.loads((Path(__file__).parents[1] / run.CONFIG).read_text())
    assert cfg["new_fits"] == 2 * len(run.NEW) == 12
    assert cfg["cached_control_fits"] == 2 * len(run.CONTROLS) == 12
    assert len(run.CONTRASTS) == 12
    assert cfg["primary"] == "scope_both"
    assert cfg["automatic_gpu_authorization"] is False


def test_secondary_winner_does_not_change_primary_gate():
    metrics, pooled, comparisons = [], [], []
    for name in run.VARIANTS:
        for fold in range(2):
            metrics.append(
                {"variant": name, "fold": fold, "auc": 0.99 if name == "scope_negation" else 0.5}
            )
        pooled.append({"variant": name, "ranked_pooled_auc": 0.5})
    for ref in (run.ANCHOR, "copy_both"):
        comparisons.append(
            {"candidate": "scope_both", "reference": ref, "delta_auc": 0, "simultaneous_low": -0.1}
        )
    cfg = {"primary": "scope_both", "minimum_macro_delta": 0.003}
    assert run.decide(metrics, comparisons, pooled, cfg)["decision"] == "DO_NOT_PROMOTE_PRIMARY"


@pytest.fixture(scope="module")
def study_template(request, tmp_path_factory):
    template_root = request.getfixturevalue("evidence_template")
    root = Path(shutil.copytree(template_root, tmp_path_factory.mktemp("r5") / "project"))
    source = Path(__file__).parents[1]
    for name in run.SOURCE_PATHS:
        if name not in run.evidence.SOURCE_PATHS:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, path)
    fourth = run.evidence.run_study(root)
    config = json.loads((root / run.CONFIG).read_text())
    config.update(
        round4_run_id=fourth["run_id"],
        round4_results_sha256=run.digest(root / run.evidence.PUBLIC / "results.json"),
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


def test_full_study_preserves_prior_evidence_and_control_parity(study_root):
    groups = ["behavioral_features", "relational_features", "policy_features", "evidence_features"]
    before = {
        str(p): run.digest(p)
        for g in groups
        for folder in ("runs", "reports")
        for p in (study_root / folder / g).rglob("*")
        if p.is_file()
    }
    result = run.run_study(study_root)
    assert result["new_fits"] == 12 and result["reused_prior_controls"] == 12
    assert result["control_design_parity"] and result["kaggle_score"] is None
    assert len(run.figures(result)) == 8
    assert all(run.digest(Path(p)) == h for p, h in before.items())
    assert all(c["query_text_in_training"] == 0 for c in result["cohorts"])
    for fold in result["scope_energy_checks"]:
        for name in ("training", "query"):
            assert all(r["max_per_term_energy_error"] < 1e-10 for r in fold[name])


def test_interrupted_candidates_reused(study_root):
    with pytest.raises(TimeoutError, match="authored interruption"):
        run.run_study(study_root, max_new_fits=3)
    result = run.run_study(study_root)
    assert result["new_fits"] == 9 and result["reused_new_fits"] == 3


def test_completed_replay_does_not_fit(study_root, monkeypatch):
    first = run.run_study(study_root)
    monkeypatch.setattr(run, "fit_candidate", lambda *a, **k: pytest.fail("unexpected fit"))
    assert first == run.run_study(study_root)


def test_query_labels_never_enter_scope_extractor(study_root, monkeypatch):
    original = sf.ScopedLexicon.transform

    def checked(self, frame):
        assert set(frame.columns) == {"body", "rule"}
        return original(self, frame)

    monkeypatch.setattr(sf.ScopedLexicon, "transform", checked)
    run.run_study(study_root)


def test_prior_source_change_blocks_fitting(study_root, monkeypatch):
    (study_root / "scripts/evidence_features.py").write_text("# changed\n")
    monkeypatch.setattr(run, "fit_candidate", lambda *a, **k: pytest.fail("unexpected fit"))
    with pytest.raises(ValueError, match="source changed"):
        run.run_study(study_root)


def test_prior_checkpoint_corruption_stops(study_root):
    cfg = json.loads((study_root / run.CONFIG).read_text())
    p = (
        study_root
        / run.evidence.PRIVATE
        / cfg["round4_run_id"]
        / "fold_0/rule_both/predictions.npz"
    )
    p.write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        run.run_study(study_root)


def test_missing_prior_marker_never_retrains(study_root):
    cfg = json.loads((study_root / run.CONFIG).read_text())
    p = study_root / run.evidence.PRIVATE / cfg["round4_run_id"] / "fold_0/rule_both/complete.json"
    p.unlink()
    with pytest.raises(ValueError, match="checkpoint missing"):
        run.run_study(study_root)


def test_current_checkpoint_corruption_stops(study_root):
    r = run.run_study(study_root)
    p = study_root / run.PRIVATE / r["run_id"] / "fold_0/scope_both/predictions.npz"
    p.write_bytes(p.read_bytes() + b"broken")
    with pytest.raises(ValueError, match="checksum"):
        run.run_study(study_root)


def test_previous_public_result_not_overwritten(study_root):
    r = run.run_study(study_root)
    p = study_root / run.PUBLIC / "results.json"
    before = p.read_bytes()
    (study_root / run.PRIVATE / r["run_id"] / "finished.json").unlink()
    c = study_root / run.CONFIG
    cfg = json.loads(c.read_text())
    cfg["bootstrap_replicates"] += 1
    c.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="different completed"):
        run.run_study(study_root)
    assert p.read_bytes() == before


def test_export_only_aggregates_and_verified_hashes(study_root, tmp_path):
    import zipfile

    run.run_study(study_root)
    p = run.export_return(study_root, tmp_path / "out.zip")
    with zipfile.ZipFile(p) as z:
        assert not any(
            n.endswith((".npz", ".joblib")) or n.startswith("data/") for n in z.namelist()
        )
        for name, sha in json.loads(z.read("SHA256SUMS.json")).items():
            assert run.hashlib.sha256(z.read(name)).hexdigest() == sha
