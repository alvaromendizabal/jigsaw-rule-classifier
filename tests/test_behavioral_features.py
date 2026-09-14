from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from scripts import behavioral_features as bf
from scripts import run_behavioral_features as run
from scripts.support_adaptation import build_study


def authored_frame():
    rows = []
    for r, rule in enumerate(
        (
            "No Advertising: referral links and promotional content are not allowed.",
            "No legal advice: Do not offer or request legal advice.",
        )
    ):
        for i in range(48):
            label = i % 2
            phrase = (
                ("Buy my product discount code" if r == 0 else "You should file a claim in court")
                if label
                else "According to this reference here is a topic discussion"
            )
            rows.append(
                {
                    "row_id": r * 100 + i,
                    "body": phrase + f" sample{r}x{i}",
                    "rule": rule,
                    "subreddit": "authored_community",
                    "rule_violation": label,
                    "positive_example_1": f"Buy our shop product promo item{r}_{i}"
                    if r == 0
                    else f"You should sue in court item{r}_{i}",
                    "positive_example_2": f"Contact me I can help offer help{r}_{i}",
                    "negative_example_1": f"Here is a factual reference info{r}_{i}",
                    "negative_example_2": f"They said the discussion happened story{r}_{i}",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "project"
    source = Path(__file__).resolve().parents[1]
    for name in run.SOURCE_PATHS:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    (root / "data/raw").mkdir(parents=True)
    frame = authored_frame()
    frame.to_csv(root / "data/raw/train.csv", index=False)
    config = json.loads((root / run.CONFIG).read_text())
    config.update(
        word_features=120,
        char_features=150,
        bootstrap_replicates=50,
        max_seconds=120,
        feature_budget_per_family=8,
    )
    return root, frame, config


def test_scope_separates_quoted_advice_from_authored_text():
    own, quote, code, meta = bf.split_scope('He said "You should sue." I disagree. `buy now`')
    assert "You should sue" not in own
    assert "You should sue" in quote
    assert "buy now" in code
    assert meta["quoted_fraction"] > 0
    assert len(own) == len(quote) == len(code)


@pytest.mark.parametrize("quoted", ['"Buy now"', "“Buy now”", "> Buy now\n"])
def test_supported_quote_forms(quoted):
    own, quote, _, _ = bf.split_scope(quoted + " Some discussion")
    assert "Buy now" in quote
    assert "Buy now" not in own


def test_fenced_code_is_not_authored():
    own, _, code, _ = bf.split_scope("```\nYou should sue\n```\nThanks")
    assert "sue" in code and "sue" not in own


def test_unmatched_quote_is_flagged_not_silently_removed():
    own, _, _, meta = bf.split_scope('"Buy now')
    assert "Buy now" in own and meta["unmatched_double_quote"] == 1


def test_contractions_are_not_single_quoted_spans():
    own, _, _, _ = bf.split_scope("I don't know")
    assert "don't" in own


def test_negation_is_sentence_local():
    names, values = bf.text_features("Do not buy now. Buy tomorrow.", "No Advertising")
    d = dict(zip(names, values, strict=True))
    assert d["scope/commercial_action_preceded_by_negation"] == pytest.approx(np.log(2))


def test_no_heuristic_assigns_labels():
    d = bf.basic_features(
        pd.DataFrame({"body": ["Not a lawyer. You should sue."], "rule": ["No legal advice"]})
    )
    assert len(d.columns) == 215
    assert "rule_violation" not in d
    assert d["behavior/disclaimer_present"].iloc[0] == 1
    assert d["behavior/directive_present"].iloc[0] == 1


@pytest.mark.parametrize("bad", [None, "", "  ", np.nan, 1])
def test_invalid_text_stops(bad):
    with pytest.raises(ValueError):
        bf.text_features(bad, "No Advertising")


def test_query_labels_not_used_in_features():
    a = authored_frame().iloc[:6]
    b = a.copy()
    b.rule_violation = 1 - b.rule_violation
    pd.testing.assert_frame_equal(bf.basic_features(a), bf.basic_features(b))


def test_global_support_exclusion_catches_other_rule_copy():
    f = authored_frame()
    f.loc[0, "positive_example_1"] = f.loc[50, "body"]
    plan, audit = run.protocol(f)
    legal = next(x for x in audit if "legal" in x["policy"])
    assert legal["readiness_same_rule_novel"] == 48
    assert legal["canonical_global_novel"] == 47
    assert legal["cross_rule_support_exclusions"] == 1
    assert 102 not in [q["row_id"] for p in plan["folds"] for q in p["queries"]]


def test_canonical_training_purges_queries_from_every_source():
    f = authored_frame()
    f.loc[0, "body"] = f.loc[50, "body"]
    plan = build_study(f)
    for fold in plan["folds"]:
        query = {bf.normalize(q["body"]) for q in fold["queries"]}
        assert not query & {bf.normalize(t["body"]) for t in fold["training"]}


def test_expected_cohort_is_a_gate():
    with pytest.raises(ValueError, match="cohort differs"):
        run.protocol(authored_frame(), [234, 647])


def test_support_reference_rejects_self_reference():
    f = authored_frame().iloc[:8]
    with pytest.raises(ValueError, match="leaked"):
        bf.support_features(f, f)


def test_support_reference_requires_same_rule_labels():
    f = authored_frame()
    with pytest.raises(ValueError, match="Same-rule"):
        bf.support_features(f.iloc[50:53], f.iloc[:10])


def test_crossfit_complete_deterministic_and_finite():
    fold = build_study(authored_frame())["folds"][0]
    t, q = pd.DataFrame(fold["training"]), pd.DataFrame(fold["queries"])
    x, v, audit = bf.cross_fitted_support(t, q)
    again, _, _ = bf.cross_fitted_support(t, q)
    pd.testing.assert_frame_equal(x, again)
    assert x.shape[1] == v.shape[1] == 65
    assert np.isfinite(x.to_numpy()).all()
    assert len(audit) == 3 and sum(a["held_rows"] for a in audit) == len(t)


def test_reference_order_invariance():
    fold = build_study(authored_frame())["folds"][0]
    t, q = pd.DataFrame(fold["training"]), pd.DataFrame(fold["queries"])
    a, b = bf.support_features(q, t), bf.support_features(q, t.iloc[::-1])
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_duplicate_constant_screen():
    x = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1], [1, 1, 1]], float)
    selected, report = bf.screen_family(x, np.array([0, 0, 1, 1]), ["a", "b", "c"], 3)
    assert selected.tolist() == [0]
    assert report["constant"] == 1


def test_checkpoint_corruption_stops(tmp_path):
    run.stage_write(tmp_path, {"test.json": b"{}"}, {"a": 1})
    (tmp_path / "test.json").write_text("broken")
    with pytest.raises(ValueError, match="checksum"):
        run.stage_read(tmp_path, {"a": 1})


def test_checkpoint_identity_stops(tmp_path):
    run.stage_write(tmp_path, {"test.json": b"{}"}, {"a": 1})
    with pytest.raises(ValueError, match="identity"):
        run.stage_read(tmp_path, {"a": 2})


def test_unfinished_checkpoint_can_complete(tmp_path):
    (tmp_path / "partial").write_text("interrupted")
    assert run.stage_read(tmp_path, {"a": 1}) is None
    run.stage_write(tmp_path, {"test.json": b"{}"}, {"a": 1})
    assert run.stage_read(tmp_path, {"a": 1})


@pytest.mark.parametrize("seed", [2, 5, 8])
def test_vectorized_tied_weighted_auc_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    y = np.arange(20) % 2
    p = rng.integers(0, 4, 20) / 4
    w = rng.integers(1, 5, (7, 20))
    actual = run.weighted_auc_samples(y, p, w)
    expected = [roc_auc_score(y, p, sample_weight=row) for row in w]
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_zero_class_bootstrap_is_explicit_nan():
    actual = run.weighted_auc_samples(np.array([0, 1]), np.array([0.2, 0.5]), np.array([[1, 0]]))
    assert np.isnan(actual[0])


def test_end_to_end_resume_and_dashboard(workspace, monkeypatch):
    root, frame, config = workspace
    first = run.run_study(root, test_frame=frame, test_config=config)
    assert first["fit_count"] == first["new_fits"] == 20
    assert len(first["comparisons"]) == 13
    assert first["new_dense_candidates"] == 280
    assert first["automatic_gpu_authorization"] is False
    assert len(run.figures(first)) == 8
    dashboard = run.write_dashboard(root, first)
    assert dashboard.is_file()
    monkeypatch.setattr(
        run.LogisticRegression,
        "fit",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not refit")),
    )
    again = run.run_study(root, test_frame=frame, test_config=config)
    assert again == first
    receipt = json.loads(
        (root / run.PRIVATE_DIR / first["run_id"] / "last_invocation.json").read_text()
    )
    assert receipt["new_fits"] == 0 and receipt["reused_fits"] == 20


def test_interrupted_run_reuses_completed_candidates(workspace):
    root, frame, config = workspace
    with pytest.raises(TimeoutError, match="interruption"):
        run.run_study(root, test_frame=frame, test_config=config, max_new_fits=2)
    result = run.run_study(root, test_frame=frame, test_config=config)
    assert result["new_fits"] == 18 and result["reused_fits"] == 2


def test_modified_public_result_stops(workspace):
    root, frame, config = workspace
    run.run_study(root, test_frame=frame, test_config=config)
    (root / run.PUBLIC_DIR / "results.json").write_text("{}")
    with pytest.raises(ValueError, match="corruption"):
        run.run_study(root, test_frame=frame, test_config=config)


def test_positive_controls_show_scope_not_ground_truth():
    pairs = pd.DataFrame(
        {"body": ["You should sue.", 'He wrote "You should sue."'], "rule": ["No legal advice"] * 2}
    )
    f = bf.basic_features(pairs)
    assert f.loc[0, "scope/directive_authored"] > f.loc[1, "scope/directive_authored"]
    assert f.loc[0, "scope/directive_quoted"] < f.loc[1, "scope/directive_quoted"]
