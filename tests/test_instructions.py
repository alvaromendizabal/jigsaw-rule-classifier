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


def test_completed_instruction_shards_do_not_load_an_encoder(tmp_path, monkeypatch):
    from jigsaw_rules import instructions

    class Encoder:
        def encode(self, texts):
            return np.tile([0.4, -0.4, 0.8], (len(texts), 1))

    monkeypatch.setattr(instructions, "_encoder", lambda *args: Encoder())
    first = instructions._shard(tmp_path, {}, tmp_path / "cache", ["a"], ["authored prompt"])

    def forbidden(*args):
        raise AssertionError("A completed shard must not load model weights")

    monkeypatch.setattr(instructions, "_encoder", forbidden)
    assert (
        instructions._shard(tmp_path, {}, tmp_path / "cache", ["a"], ["authored prompt"]) == first
    )


def test_cached_prompt_preparation_needs_only_verified_tokenizer_assets(tmp_path, monkeypatch):
    import hashlib

    from jigsaw_rules.instructions import prepare_instruction_model

    path = tmp_path / "models/qwen3-instruction/authored"
    path.mkdir(parents=True)
    payload = b"authored tokenizer fixture"
    (path / "tokenizer.json").write_bytes(payload)
    config = {
        "revision": "authored",
        "files": {
            "tokenizer.json": {
                "algorithm": "sha256",
                "digest": hashlib.sha256(payload).hexdigest(),
            },
            "model.safetensors": {"algorithm": "sha256", "digest": "0" * 64},
        },
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("No model download is needed to restore complete features")

    monkeypatch.setattr("jigsaw_rules.instructions.subprocess.run", forbidden)
    assert prepare_instruction_model(tmp_path, config, weights=False) == path
