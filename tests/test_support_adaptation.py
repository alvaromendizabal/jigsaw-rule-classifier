import copy
import json

import pandas as pd
import pytest

from jigsaw_rules.data import synthetic
from scripts.decision_training import decision_loss, train_adapter
from scripts.support_adaptation import (
    add_adapter,
    build_study,
    decision_prompts,
    validate_study,
)
from tests.test_competition_features import ChatTokenizer, spec


def test_query_labels_cannot_change_their_adaptation_pool(tmp_path):
    synthetic(tmp_path)
    frame = pd.read_csv(tmp_path / "train.csv")
    plan = build_study(frame)
    query_ids = [row["row_id"] for row in plan["folds"][0]["queries"]]
    frame.loc[frame.row_id.isin(query_ids), "rule_violation"] ^= 1
    assert build_study(frame)["folds"][0] == plan["folds"][0]
    corrupted = copy.deepcopy(plan)
    corrupted["folds"][0]["queries"][0]["rule_violation"] = 1
    with pytest.raises(ValueError, match="exclude targets"):
        validate_study(corrupted)
    corrupted = copy.deepcopy(plan)
    corrupted["folds"][0]["training"][0]["body"] = corrupted["folds"][0]["queries"][0]["body"]
    with pytest.raises(ValueError, match="evaluation text"):
        validate_study(corrupted)


def test_support_overlap_is_excluded_independently_of_its_label(tmp_path):
    synthetic(tmp_path)
    frame = pd.read_csv(tmp_path / "train.csv")
    frame.loc[0, "body"] = frame.loc[0, "positive_example_1"]
    plan = build_study(frame)
    assert 0 not in {row["row_id"] for fold in plan["folds"] for row in fold["queries"]}
    assert sum(fold["audit"]["known_support_overlap_excluded"] for fold in plan["folds"]) == 1


def test_matched_decision_template(tmp_path):
    from scripts.competition_features import prepare_prompts

    synthetic(tmp_path)
    frame = pd.read_csv(tmp_path / "test.csv")
    full, _ = prepare_prompts(frame, ChatTokenizer(), spec())
    assert decision_prompts(frame.to_dict("records"), ChatTokenizer(), spec()) == full[::3]


class TinyTokenizer:
    def encode(self, text, **kwargs):
        return [2] if text == "No" else [3] if text == "Yes" else [1 + ord(c) % 29 for c in text]

    def __call__(self, texts, **kwargs):
        import torch

        values = [self.encode(text) for text in texts]
        size = max(map(len, values))
        ids = torch.tensor([[0] * (size - len(row)) + row for row in values])
        return {"input_ids": ids, "attention_mask": ids.ne(0).long()}


def training_spec():
    return {
        "rank": 2,
        "alpha": 4,
        "dropout": 0.1,
        "target_modules": ["q_proj", "v_proj"],
        "seed": 91,
        "effective_batch": 4,
        "micro_batch": 2,
        "learning_rate": 0.01,
        "weight_decay": 0.01,
        "warmup_fraction": 0.1,
        "gradient_clip": 1.0,
        "gradient_checkpointing": True,
        "checkpoint_steps": 1,
    }


def tiny_model():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(91)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
    )
    return add_adapter(Qwen3ForCausalLM(config), training_spec())


class SilentLog:
    def emit(self, *args, **kwargs):
        pass


def test_optimizer_recovery_matches_uninterrupted_training(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from peft import get_peft_model_state_dict

    torch.set_num_threads(2)
    args = {
        "tokenizer": TinyTokenizer(),
        "texts": [f"authored {i}" for i in range(8)],
        "labels": [0, 1] * 4,
        "repeats": [1] * 8,
        "spec": training_spec(),
        "log": SilentLog(),
    }
    full = tiny_model()
    train_adapter(full, output=tmp_path / "full", **args)
    interrupted = tiny_model()

    def stop(path):
        if path.name == "complete.json":
            raise InterruptedError("Controlled interruption")

    with pytest.raises(InterruptedError):
        train_adapter(interrupted, output=tmp_path / "resumed", sync=stop, **args)
    resumed = tiny_model()
    record = train_adapter(resumed, output=tmp_path / "resumed", **args)
    assert record["resumed_step"] == 1
    for name, actual in get_peft_model_state_dict(resumed, save_embedding_layers=False).items():
        expected = get_peft_model_state_dict(full, save_embedding_layers=False)[name]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    marker = tmp_path / "resumed/step_000002/complete.json"
    manifest = json.loads(marker.read_text())
    manifest["sha256"] = "0" * 64
    marker.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        train_adapter(tiny_model(), output=tmp_path / "resumed", **args)


def test_peft_decision_representation_and_loss_use_last_token():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    model = tiny_model().eval()
    inputs = TinyTokenizer()(["a", "authored"])
    targets = torch.tensor([2, 3])
    base = model.get_base_model()
    with torch.no_grad():
        expected = model(**inputs, use_cache=False).logits[:, -1]
        hidden = base.model(**inputs, use_cache=False).last_hidden_state[:, -1]
        torch.testing.assert_close(base.lm_head(hidden), expected)
        torch.testing.assert_close(
            decision_loss(model, inputs, targets),
            torch.nn.functional.cross_entropy(expected, targets),
        )
