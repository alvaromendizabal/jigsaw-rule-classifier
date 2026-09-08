import json
from pathlib import Path

import numpy as np
import pytest

from jigsaw_rules.instructions import instruction_candidates, instruction_inputs
from tests.test_research import frame


class Tokenizer:
    def encode(self, text, **kwargs):
        return text.split()

    def decode(self, tokens, **kwargs):
        return " ".join(tokens)

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return json.dumps(messages)


def test_instruction_prompts_ignore_labels_and_support_order():
    config = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/instructions.json").read_text()
    )
    data = frame(2)
    first, _ = instruction_inputs(data, Tokenizer(), config)
    data.rule_violation = 1 - data.rule_violation
    data["positive_example_1"], data["positive_example_2"] = (
        data.positive_example_2.copy(),
        data.positive_example_1.copy(),
    )
    second, _ = instruction_inputs(data, Tokenizer(), config)
    assert first == second
    assert len(first) == 6
    assert "community_rule" in first[0] and "violating_examples" not in first[0]
    assert "community_rule" not in first[1] and "violating_examples" in first[1]


def test_instruction_candidates_have_finite_schema_and_exact_counts():
    x, names = instruction_candidates(np.ones((20, 3, 3)))
    assert x.shape == (20, 36)
    assert len(set(names)) == 36
    with pytest.raises(ValueError, match="schema"):
        instruction_candidates(np.full((2, 3, 3), np.nan))
