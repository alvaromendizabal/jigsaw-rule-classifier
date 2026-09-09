"""Selected-feature parity, target isolation, routing and artifact integrity."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from jigsaw_rules.data import synthetic
from jigsaw_rules.diagnostics import support_scores
from jigsaw_rules.final_model import FeaturePipeline, RoutedClassifier, familiar_mask, verify_stage
from jigsaw_rules.representations import semantic_candidates, structural_candidates
from jigsaw_rules.research import CORE, _model, fold_bank
from jigsaw_rules.runtime import stage


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    directory = tmp_path_factory.mktemp("selected_features")
    synthetic(directory / "data")
    frame = pd.read_csv(directory / "data/train.csv")
    rng = np.random.default_rng(55)
    vectors = rng.normal(size=(len(frame), 5, 8))
    vectors /= np.linalg.norm(vectors, axis=2, keepdims=True)
    training, valid = frame.iloc[:72], frame.iloc[72:]
    structural, sn = structural_candidates(frame)
    semantic, en = semantic_candidates(vectors)
    banks = fold_bank(
        directory,
        "authored",
        training,
        valid,
        (structural[:72], structural[72:], sn),
        (semantic[:72], semantic[72:], en),
        lambda: None,
    )
    stage(
        directory,
        "authored_model_all_transfer",
        lambda p: _model(p, training, valid, banks, CORE, 0),
    )
    loaded = FeaturePipeline.from_legacy(directory, "authored", training, 8)
    return directory, frame, vectors, loaded, banks


def test_portable_transform_matches_original_feature_banks_and_predictions(fitted):
    directory, frame, vectors, model, banks = fitted
    inputs = frame.iloc[72:].drop(columns="rule_violation")
    expected = sparse.hstack([sparse.load_npz(banks[n] / "valid.npz") for n in CORE], format="csr")
    actual = model.transform(inputs, vectors[72:])
    np.testing.assert_allclose(actual.toarray(), expected.toarray(), rtol=0, atol=1e-12)
    saved = pd.read_csv(directory / "authored_model_all_transfer/predictions.csv")
    np.testing.assert_allclose(model.predict(inputs, vectors[72:]), saved.probability, atol=1e-12)


def test_new_fit_matches_the_existing_fixed_seven_family_pipeline(fitted):
    _, frame, vectors, original, _ = fitted
    spec = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/model_validation.json").read_text()
    )
    new = FeaturePipeline().fit(frame.iloc[:72], vectors[:72], spec["classifier"])
    inputs = frame.iloc[72:].drop(columns="rule_violation")
    assert new.selected_ == original.selected_
    np.testing.assert_allclose(
        new.predict(inputs, vectors[72:]), original.predict(inputs, vectors[72:]), atol=1e-12
    )


def test_prediction_is_target_free_serializable_and_batch_independent(fitted, tmp_path):
    _, frame, vectors, model, _ = fitted
    inputs, v = frame.iloc[72:].drop(columns="rule_violation"), vectors[72:]
    with pytest.raises(ValueError, match="target labels"):
        model.predict(frame.iloc[72:], v)
    with pytest.raises(ValueError, match="dimension"):
        model.predict(inputs, v[:, :, :4])
    p = model.predict(inputs, v)
    np.testing.assert_allclose(
        model.predict(inputs.iloc[[4, 0, 2]], v[[4, 0, 2]]), p[[4, 0, 2]], atol=1e-12
    )
    file = tmp_path / "model.joblib"
    joblib.dump(model, file)
    np.testing.assert_array_equal(joblib.load(file).predict(inputs, v), p)


def test_new_policies_bypass_all_learned_feature_transforms(fitted, monkeypatch):
    _, frame, vectors, model, _ = fitted
    inputs = frame.iloc[72:].drop(columns="rule_violation").copy()
    inputs["rule"] = "A previously unseen policy"
    monkeypatch.setattr(
        model, "predict", lambda *a: pytest.fail("Unseen route used learned features")
    )
    actual = RoutedClassifier(model).predict(inputs, vectors[72:])
    np.testing.assert_array_equal(actual, support_scores(vectors[72:])["qwen_centroid"])
    assert not familiar_mask(inputs, model.policies_).any()


def test_policy_membership_is_normalized_and_bound_to_actual_parent_training(fitted):
    directory, frame, _, model, _ = fitted
    inputs = frame.iloc[:3].drop(columns="rule_violation").copy()
    inputs["rule"] = "  " + inputs.rule.str.upper() + "  "
    assert familiar_mask(inputs, model.policies_).all()
    with pytest.raises(ValueError, match="training identities"):
        FeaturePipeline.from_legacy(directory, "authored", frame.iloc[1:73], 8)


def test_corrupt_and_escaping_artifacts_are_rejected(tmp_path):
    (tmp_path / "complete.json").write_text(json.dumps({"files": {"../outside": "bad"}}))
    with pytest.raises(ValueError, match="path"):
        verify_stage(tmp_path)
    (tmp_path / "item").write_text("tampered")
    (tmp_path / "complete.json").write_text(json.dumps({"files": {"item": "bad"}}))
    with pytest.raises(ValueError, match="checksum"):
        verify_stage(tmp_path)
