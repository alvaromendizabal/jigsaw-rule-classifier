"""Scope preservation, target isolation, cache integrity and complete resumable ablations."""

import shutil
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from jigsaw_rules import formatting
from jigsaw_rules.data import EXAMPLES
from jigsaw_rules.diagnostics import support_scores
from jigsaw_rules.embeddings import QwenEncoder, load_spec
from jigsaw_rules.expanded import design
from jigsaw_rules.runtime import atomic_bytes, atomic_json, digest, stage
from tests.test_research import frame, vectors

ROOT = Path(__file__).resolve().parents[1]
RULES = [
    "No Advertising: Spam, referral links, unsolicited advertising, "
    "and promotional content are not allowed.",
    "No legal advice: Do not offer or request legal advice.",
    "No medical advice: Do not offer or request specific medical advice, diagnoses, "
    "or treatment recommendations.",
    "No promotion of illegal activity: Do not encourage or promote illegal activities, "
    "such as drug-related activity, violence, exploitation, theft, or other criminal behavior.",
]


def cohort():
    data = frame(120)
    data["rule"] = [RULES[i // 30] for i in range(len(data))]
    return data


def test_behavior_compiler_preserves_offer_request_and_full_policy_scope():
    assert formatting.behavior_hypothesis(RULES[1]) == (
        "The author uses this comment to offer or request legal advice."
    )
    for rule in RULES[1:]:
        scope = rule.split("Do not ")[1]
        assert formatting.behavior_hypothesis(rule) == "The author uses this comment to " + scope
    assert formatting.behavior_hypothesis(RULES[0]) == (
        "This comment contains spam, referral links, unsolicited advertising, "
        "and promotional content."
    )
    with pytest.raises(ValueError, match="Unsupported"):
        formatting.behavior_hypothesis("Do the right thing")
    with pytest.raises(ValueError, match="Unsupported"):
        formatting.behavior_hypothesis("Advice: some advice may be allowed")


def test_inputs_exclude_targets_and_do_not_conflate_body_and_document_roles():
    data = cohort().iloc[:3]
    for function in (formatting.intent_inputs, formatting.document_inputs):
        with pytest.raises(ValueError, match="query targets"):
            function(data)
    query = data.drop(columns="rule_violation")
    pairs = formatting.intent_inputs(query)
    assert len(pairs) == 6
    assert all(p[0] == data.iloc[i // 2].body for i, p in enumerate(pairs))
    assert formatting.document_inputs(query) == query[EXAMPLES].to_numpy().ravel().tolist()
    assert formatting.document_encoder(ROOT).contract != QwenEncoder(ROOT, load_spec(ROOT)).contract


def test_semantic_geometry_support_order_batch_and_immutability():
    original, docs = vectors(20), vectors(20)[:, 1:]
    before = original.copy()
    combined = formatting.asymmetric_vectors(original, docs)
    np.testing.assert_array_equal(combined[:, 0], original[:, 0])
    np.testing.assert_array_equal(original, before)
    expected = support_scores(combined)["qwen_centroid"]
    np.testing.assert_allclose(
        expected, support_scores(combined[:, [0, 2, 1, 4, 3]])["qwen_centroid"]
    )
    np.testing.assert_array_equal(expected[:5], support_scores(combined[:5])["qwen_centroid"])
    with pytest.raises(ValueError):
        formatting.asymmetric_vectors(original, docs * 2)


def test_nli_feature_schema_and_batch_independence():
    p = np.random.default_rng(3).dirichlet([2, 3, 4], size=(20, 2))
    matrix, names = formatting.intent_features(p)
    assert matrix.shape == (20, 45)
    assert len(names) == len(set(names)) == 45
    np.testing.assert_array_equal(formatting.intent_features(p[:5])[0], matrix[:5])
    p[0, 0, 1] = 2
    with pytest.raises(ValueError):
        formatting.intent_features(p)


def test_training_screen_does_not_read_validation_labels_and_roundtrips(tmp_path):
    data, raw = cohort(), np.random.default_rng(5).normal(size=(120, 45))
    names = [str(i) for i in range(45)]
    query = data.iloc[90:].copy()
    query["rule_violation"] = "MUST_NOT_INTERPRET"
    path = stage(
        tmp_path,
        "bank",
        partial(
            formatting.build_bank,
            training=data.iloc[:90],
            validation=query,
            raw_train=raw[:90],
            raw_valid=raw[90:],
            names=names,
            budget=16,
        ),
    )
    screen, scaler = joblib.load(path / "screen.joblib")
    expected = scaler.transform(screen.transform(raw[90:], names)) / np.sqrt(len(screen.indices_))
    np.testing.assert_array_equal(sparse.load_npz(path / "valid.npz").toarray(), expected)
    assert len(screen.indices_) == 16


def test_probability_cache_checks_members_and_rejects_corruption(tmp_path):
    def save(path):
        np.save(path / "probabilities.npy", [[0.1, 0.7, 0.2]])
        atomic_json(path / "inputs.json", ["key"])

    path = stage(tmp_path, "batch_one", save)
    found, _ = formatting.probability_shards(tmp_path)
    np.testing.assert_array_equal(found["key"], [0.1, 0.7, 0.2])
    (path / "probabilities.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        formatting.probability_shards(tmp_path)


def test_error_sampling_is_balanced_order_independent_and_not_extreme_only():
    data = cohort()
    plan = formatting.load_plan(ROOT)
    p = np.linspace(0.01, 0.99, len(data))
    sample = formatting.sample_errors(data, p, plan)
    assert len(sample) == 48
    assert sample.groupby(["rule", "rule_violation", "score_tertile"]).size().eq(4).all()
    order = np.random.default_rng(2).permutation(len(data))
    shuffled = formatting.sample_errors(data.iloc[order], p[order], plan)
    assert sample.row_id.tolist() == shuffled.row_id.tolist()


def test_complete_study_reuses_banks_and_never_refits_completed_stages(tmp_path, monkeypatch):
    data, original = cohort(), vectors(120)
    documents = vectors(120)[:, [4, 3, 2, 1]]
    p = np.random.default_rng(8).dirichlet([2, 3, 4], size=(120, 2))
    plan = formatting.load_plan(ROOT)
    plan["expanded_run"], plan["bootstrap_draws"] = "software_test", 100
    monkeypatch.setattr(formatting, "load_plan", lambda root: plan)
    monkeypatch.setattr(formatting, "load_development", lambda root: data)
    monkeypatch.setattr(
        formatting, "expanded_evidence", lambda root: {"metadata": {"run_id": "software_test"}}
    )
    monkeypatch.setattr(
        formatting, "cached_vectors", lambda root, frame: (original, {"test": True})
    )
    monkeypatch.setattr(
        formatting, "plain_support_vectors", lambda *a, **k: (documents, {"test": True})
    )
    monkeypatch.setattr(formatting, "intent_probabilities", lambda *a, **k: (p, {"test": True}))
    for relative in (
        "configs/formatting.json",
        "src/jigsaw_rules/pairs.py",
        "reports/formatting/error_audit.json",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    atomic_json(tmp_path / "reports/expanded/metadata.json", {"test": True})
    base = tmp_path / "runs/expanded/software_test"
    splits = design(data, {"seen_rule_folds": 2, "seed": 2025})

    def save_design(path):
        atomic_json(path / "splits.json", splits)
        atomic_json(path / "row_ids.json", data.row_id.tolist())

    stage(base, "design", save_design)
    controls = []
    raw = np.random.default_rng(20).normal(size=(120, 40))
    for protocol, folds in splits.items():
        for fold, assignment in enumerate(folds):
            ti, vi = assignment["train"], assignment["valid"]
            for family in ("word", "semantic_scalar"):
                stage(
                    base,
                    f"{protocol}_{fold}_{family}",
                    partial(
                        formatting.build_bank,
                        training=data.iloc[ti],
                        validation=data.iloc[vi],
                        raw_train=raw[ti],
                        raw_valid=raw[vi],
                        names=[str(i) for i in range(40)],
                        budget=16,
                    ),
                )
        for model in ("qwen_centroid", "semantic_scalar_only", "word_semantic_scalar"):
            part = data[["row_id", "rule", "rule_violation"]].copy()
            assignment = {
                data.iloc[i].row_id: fold for fold, rows in enumerate(folds) for i in rows["valid"]
            }
            controls.append(
                part.assign(
                    protocol=protocol,
                    model=model,
                    fold=part.row_id.map(assignment),
                    probability=support_scores(original)["qwen_centroid"],
                )
            )
    stage(
        base,
        "review",
        lambda path: atomic_bytes(
            path / "oof.csv", pd.concat(controls).to_csv(index=False).encode()
        ),
    )
    result = formatting.run_formatting(tmp_path)
    evidence = formatting.formatting_evidence(tmp_path)
    assert evidence["audit"]["fitted_models"] == 24
    assert evidence["audit"]["candidate_counts"] == {"asymmetric_scalar": 32, "intent_scalar": 45}
    oof = pd.read_csv(result / "review/oof.csv")
    assert oof.groupby(["protocol", "model"]).size().eq(120).all()
    from scripts import verify_formatting

    checked = verify_formatting.verify(tmp_path)
    assert checked["metric_records"] == 20
    assert checked["model_prediction_replays"] == 24
    assert checked["feature_matrix_replays"] == 24
    assert checked["frozen_score_replays"] == 8
    assert checked["new_encoder_calls"] == checked["new_model_fits"] == 0
    from scripts import build_formatting_report

    (tmp_path / "scripts").mkdir(exist_ok=True)
    shutil.copyfile(
        ROOT / "scripts/build_formatting_report.py", tmp_path / "scripts/build_formatting_report.py"
    )
    build_formatting_report.build(tmp_path, synthetic_fixture=True)
    assert len(build_formatting_report.verify_figures(tmp_path)["files"]) == 4
    with pytest.raises(ValueError, match="Synthetic"):
        build_formatting_report.display_figure(tmp_path, "contrasts")
    markers = {p: digest(p) for p in result.glob("*/complete.json")}

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed fit was repeated")

    monkeypatch.setattr(formatting, "_model", forbidden)
    monkeypatch.setattr(formatting, "build_bank", forbidden)
    assert formatting.run_formatting(tmp_path) == result
    assert all(digest(path) == sha for path, sha in markers.items())
    (tmp_path / "reports/formatting/results.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        formatting.formatting_evidence(tmp_path)
