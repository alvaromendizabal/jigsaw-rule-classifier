import json
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_backbone_capacity import blend_scores, promotion_decision
from scripts.model_study import DecisionTokenizer


def test_native_chat_options_cannot_be_silently_overridden():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages, kwargs

        def encode(self, text, **kwargs):
            return [len(text)]

        def decode(self, ids, **kwargs):
            return str(ids)

        def __call__(self, texts, **kwargs):
            return texts

    tokenizer = DecisionTokenizer(Tokenizer(), {"enable_thinking": False})
    messages, options = tokenizer.apply_chat_template(["authored"], tokenize=False)
    assert messages == ["authored"] and options["enable_thinking"] is False
    assert tokenizer.encode("No") == [2] and tokenizer(["a"]) == ["a"]
    assert tokenizer.decode([3]) == "[3]"
    with pytest.raises(ValueError, match="override"):
        tokenizer.apply_chat_template(["authored"], enable_thinking=True)


def test_only_supported_registered_candidates_can_be_selected():
    results = {
        "qwen4b": {"rule_macro_auc": 0.7, "per_rule_auc": {"a": 0.6, "b": 0.8}},
        "qwen8b": {"rule_macro_auc": 0.75, "per_rule_auc": {"a": 0.7, "b": 0.8}},
        "blend": {"rule_macro_auc": 0.76, "per_rule_auc": {"a": 0.71, "b": 0.81}},
        "qwen8b_frozen": {"rule_macro_auc": 0.99, "per_rule_auc": {"a": 0.99, "b": 0.99}},
    }
    intervals = [{"candidate": name, "simultaneous_lower": 0.01} for name in ("qwen8b", "blend")]
    gate = {"simultaneous_ci_lower_minimum": 0, "maximum_per_policy_regression": 0}
    selection = {"candidates": ["qwen8b", "blend"], "priority": ["qwen8b", "blend"]}
    assert promotion_decision(results, intervals, gate, selection)["selected"] == "blend"
    intervals[1]["simultaneous_lower"] = -0.01
    assert promotion_decision(results, intervals, gate, selection)["selected"] == "qwen8b"
    results["qwen8b"]["per_rule_auc"]["b"] = 0.79
    assert promotion_decision(results, intervals, gate, selection)["selected"] is None


def test_fixed_blend_rejects_misaligned_or_nonfinite_arrays():
    weights = {"qwen4b": 0.5, "qwen8b": 0.5}
    np.testing.assert_array_equal(blend_scores([0, 2], [2, 0], weights), [0.5, 0.5])
    with pytest.raises(ValueError, match="aligned"):
        blend_scores([0], [0, 1], weights)
    with pytest.raises(ValueError, match="finite"):
        blend_scores([np.inf], [0], weights)


def test_capacity_protocol_preserves_original_plan_and_predeadline_assets():
    root = Path(__file__).resolve().parents[1]
    capacity = json.loads((root / "configs/backbone_capacity.json").read_text())
    original = json.loads((root / "configs/support_adaptation.json").read_text())
    model = json.loads((root / "configs/qwen3_8b.json").read_text())
    assert capacity["plan_sha256"] == original["plan_sha256"]
    assert capacity["training"]["effective_batch"] == original["training"]["effective_batch"]
    assert model["revision_date"] < "2025-10-23" and len(model["files"]) == 13
    assert model["chat_template_kwargs"] == {"enable_thinking": False}


def test_authored_gpu_probe_sources_match_the_generator():
    import nbformat

    from scripts.build_capacity_probe import build

    root = Path(__file__).resolve().parents[1]
    actual = nbformat.read(root / "kaggle/capacity_probe.ipynb", as_version=4)
    expected = build(root)
    assert [(c.cell_type, c.source) for c in actual.cells] == [
        (c.cell_type, c.source) for c in expected.cells
    ]
