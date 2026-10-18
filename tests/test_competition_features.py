import json
from pathlib import Path

import pytest

from scripts.competition_features import (
    prepare_prompts,
    score_logits,
    support_pairs,
)
from tests.fixtures import Tokenizer, frame


def spec():
    return json.loads(
        (Path(__file__).resolve().parents[1] / "configs/competition_features.json").read_text()
    )


class ChatTokenizer(Tokenizer):
    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages)


def test_target_free_prompts_preserve_support_order_and_decision():
    data = frame(3).drop(columns="rule_violation")
    first, _ = prepare_prompts(data, ChatTokenizer(), spec())
    data["positive_example_1"], data["positive_example_2"] = (
        data.positive_example_2.copy(),
        data.positive_example_1.copy(),
    )
    second, _ = prepare_prompts(data, ChatTokenizer(), spec())
    assert first == second
    assert "community_rule" in first[0] and "violating_examples" not in first[0]
    assert "community_rule" not in first[1] and "violating_examples" in first[1]
    assert "community_rule" in first[2] and "violating_examples" in first[2]
    with pytest.raises(ValueError, match="target"):
        prepare_prompts(frame(3), ChatTokenizer(), spec())


def test_augmentation_excludes_available_targets_conflicts_and_validation_text():
    train = frame(3)
    available = frame(3).drop(columns="rule_violation")
    available.loc[0, "positive_example_1"] = "Conflicting text"
    available.loc[0, "negative_example_1"] = "conflicting   TEXT"
    available.loc[1, "positive_example_1"] = "Forbidden validation text"
    pairs, audit = support_pairs(train, available, forbidden=["forbidden validation TEXT"])
    assert "conflicting text" not in set(pairs.normalized_body)
    assert "forbidden validation text" not in set(pairs.normalized_body)
    assert audit["conflicting_pairs"] >= 1
    assert audit["forbidden_text_occurrences"] == 1
    assert len(pairs) == len(pairs.drop_duplicates(["normalized_rule", "normalized_body"]))
    with pytest.raises(ValueError, match="target"):
        support_pairs(train, frame(3))
    reversed_pairs, _ = support_pairs(
        train.iloc[::-1], available.iloc[::-1], forbidden=["forbidden validation TEXT"]
    )
    assert pairs.equals(reversed_pairs)


def test_answer_mass_is_separate_from_binary_probability():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 8.0]])
    result = score_logits(logits, [[0, 1], [2, 3]])
    assert result[0, 0] == 0.5
    assert result[0, 1] == 0.0
    assert result[0, 2] < 0.01  # Most mass lies on a non-answer token.


def test_last_token_hidden_projection_matches_causal_lm():
    torch = pytest.importorskip("torch")
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(2025)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
    )
    model = Qwen3ForCausalLM(config).eval()
    ids = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 4]])
    mask = ids.ne(0).long()
    with torch.inference_mode():
        expected = model(ids, attention_mask=mask, use_cache=False).logits[:, -1]
        hidden = model.model(ids, attention_mask=mask, use_cache=False).last_hidden_state[:, -1]
        actual = model.lm_head(hidden)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(score_logits(actual, [[1], [2]]), score_logits(expected, [[1], [2]]))
