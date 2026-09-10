import json
from pathlib import Path

import numpy as np
import pytest

from jigsaw_rules.runtime import atomic_json, digest
from scripts.complementary_worker import checked_fold, restore_keys, verified_plan
from scripts.evaluate_complementarity import blend_scores, promotion_decision

ROOT = Path(__file__).resolve().parents[1]


def test_blend_preserves_ties_and_ignores_monotone_model_scale():
    qwen = np.array([-200.0, -200.0, 10.0, 150.0])
    phi = np.array([1.0, 3.0, 2.0, 4.0])
    weights = {"qwen": 0.5, "phi": 0.5}
    expected = np.array([0.1875, 0.4375, 0.5, 0.875])
    np.testing.assert_array_equal(blend_scores(qwen, phi, weights), expected)
    np.testing.assert_array_equal(blend_scores(qwen * 4 + 7, phi**3, weights), expected)
    for left, right in (([1, 2], [1]), ([np.nan], [1]), ([], [])):
        with pytest.raises(ValueError):
            blend_scores(left, right, weights)
    with pytest.raises(ValueError, match="sum"):
        blend_scores(qwen, phi, {"qwen": 0.6, "phi": 0.6})


def test_completed_fold_rejects_wrong_order_even_with_valid_file_hash(tmp_path):
    fold = {"rule": "rule", "queries": [{"row_id": 3}, {"row_id": 9}]}
    artifact = tmp_path / "representations.npz"
    np.savez(
        artifact,
        query_row_ids=[9, 3],
        query_frozen_scores=np.zeros((2, 3)),
        query_adapted_scores=np.ones((2, 3)),
    )
    atomic_json(
        tmp_path / "complete.json",
        {
            "fold": 0,
            "rule": "rule",
            "sha256": digest(artifact),
        },
    )
    with pytest.raises(ValueError, match="order"):
        checked_fold(tmp_path, 0, fold)
    with pytest.raises(ValueError, match="identity"):
        checked_fold(tmp_path, 1, fold)


def test_checkpoint_restore_skips_uncommitted_and_old_states():
    keys = ["run/contract.json", "run/fold_0/representations.npz"]
    for fold in (0, 1):
        for step in (8, 16, 24):
            keys.extend(
                f"run/fold_{fold}/training/step_{step:06d}/{name}"
                for name in ("state.pt", "complete.json")
            )
    keys.append("run/fold_0/training/step_000032/state.pt")
    result = restore_keys(keys)
    assert "run/contract.json" in result
    assert not any("000008" in k or "000032" in k for k in result)
    assert len(result) == 10


def test_promotion_requires_uncertainty_and_each_policy_gain():
    spec = {"simultaneous_ci_lower_minimum": 0, "maximum_per_policy_regression": 0}
    results = {
        "qwen": {"rule_macro_auc": 0.7, "per_rule_auc": {"a": 0.6, "b": 0.8}},
        "blend": {"rule_macro_auc": 0.72, "per_rule_auc": {"a": 0.63, "b": 0.81}},
    }
    uncertainty = [{"candidate": "blend", "simultaneous_lower": 0.001}]
    assert promotion_decision(results, uncertainty, spec)["eligible_for_kaggle_candidate"]
    uncertainty[0]["simultaneous_lower"] = -0.001
    assert not promotion_decision(results, uncertainty, spec)["eligible_for_kaggle_candidate"]
    uncertainty[0]["simultaneous_lower"] = 0.001
    results["blend"]["per_rule_auc"]["b"] = 0.79
    assert not promotion_decision(results, uncertainty, spec)["eligible_for_kaggle_candidate"]


def test_plan_checks_hash_and_query_label_boundary(tmp_path):
    import pandas as pd

    from jigsaw_rules.data import synthetic
    from scripts.support_adaptation import build_study

    synthetic(tmp_path)
    plan = build_study(pd.read_csv(tmp_path / "train.csv"))
    path = tmp_path / "plan.json"
    atomic_json(path, plan)
    spec = {"plan_sha256": digest(path)}
    assert verified_plan(path, spec) == plan
    plan["folds"][0]["queries"][0]["rule_violation"] = 1
    atomic_json(path, plan)
    with pytest.raises(ValueError, match="checksum"):
        verified_plan(path, spec)
    with pytest.raises(ValueError, match="exclude targets"):
        verified_plan(path, {"plan_sha256": digest(path)})


def test_native_phi_fused_adapter_optimizer_recovery_is_exact(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from peft import get_peft_model_state_dict
    from transformers import Phi3Config, Phi3ForCausalLM

    from scripts.decision_training import train_adapter
    from scripts.support_adaptation import add_adapter
    from tests.test_support_adaptation import SilentLog, TinyTokenizer, training_spec

    torch.set_num_threads(2)
    training = {
        **training_spec(),
        "target_modules": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        "keep_checkpoints": 2,
    }

    def model():
        torch.manual_seed(91)
        config = Phi3Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=512,
            original_max_position_embeddings=512,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        return add_adapter(Phi3ForCausalLM(config), training)

    args = {
        "tokenizer": TinyTokenizer(),
        "texts": [f"comment {i}" for i in range(8)],
        "labels": [0, 1] * 4,
        "repeats": [1] * 8,
        "spec": training,
        "log": SilentLog(),
    }
    full = model()
    train_adapter(full, output=tmp_path / "full", **args)

    def stop(path):
        if path.name == "complete.json":
            raise InterruptedError()

    with pytest.raises(InterruptedError):
        train_adapter(model(), output=tmp_path / "resumed", sync=stop, **args)
    resumed = model()
    record = train_adapter(resumed, output=tmp_path / "resumed", **args)
    assert record["resumed_step"] == 1
    expected = get_peft_model_state_dict(full, save_embedding_layers=False)
    for key, value in get_peft_model_state_dict(resumed, save_embedding_layers=False).items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def test_experiment_uses_predeadline_pin_and_fixed_baseline():
    spec = json.loads((ROOT / "configs/complementarity.json").read_text())
    model = json.loads((ROOT / "configs/complementary_model.json").read_text())
    assert len(model["revision"]) == 40 and model["revision_date"] < "2025-10-23"
    assert model["model_id"] == "microsoft/Phi-4-mini-instruct"
    assert spec["blend_weights"] == {"qwen": 0.5, "phi": 0.5}
    assert len(spec["baseline"]["fold_sha256"]) == 2
    assert len(model["files"]) == 11
