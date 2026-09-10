import copy
import json

import pytest

from scripts.support_context import context_prompts, retrieve


def fold():
    return {
        "rule": "No advertising",
        "queries": [
            {"row_id": 1, "body": "discount guitar for sale today", "rule": "No advertising"}
        ],
        "training": [
            {
                "body": "discount guitar for sale",
                "rule": "No advertising",
                "rule_violation": 1,
                "repeat": 2,
            },
            {
                "body": "my guitar sounds great",
                "rule": "No advertising",
                "rule_violation": 0,
                "repeat": 2,
            },
            {
                "body": "free legal consultation",
                "rule": "No advertising",
                "rule_violation": 1,
                "repeat": 2,
            },
            {
                "body": "discount guitar for sale today",
                "rule": "No other policy",
                "rule_violation": 0,
                "repeat": 1,
            },
        ],
    }


def clean_fold():
    result = fold()
    result["training"] = result["training"][:-1]
    return result


def test_query_body_is_rejected_even_under_another_policy():
    with pytest.raises(ValueError, match="evaluation text"):
        retrieve(fold())


def test_retrieval_is_balanced_relevant_and_independent_of_source_order():
    current = clean_fold()
    expected = [
        {
            "violating_examples": ["discount guitar for sale"],
            "permitted_examples": ["my guitar sounds great"],
        }
    ]
    assert retrieve(current) == expected
    current["training"].reverse()
    assert retrieve(current) == expected


def test_query_targets_are_forbidden():
    current = clean_fold()
    current["queries"][0]["rule_violation"] = 1
    with pytest.raises(ValueError, match="exclude targets"):
        retrieve(current)


def test_support_class_cannot_be_borrowed_from_another_rule():
    current = clean_fold()
    current["training"][1]["rule"] = "Another rule"
    with pytest.raises(ValueError, match="Both eligible"):
        retrieve(current)


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text)

    def decode(self, ids, **kwargs):
        return "".join(ids)

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return json.dumps(messages)


def test_native_prompt_preserves_query_and_both_explicit_labels():
    tokenizer = Tokenizer()
    spec = {"field_tokens": {"body": 384, "rule": 96, "support": 192}, "max_tokens": 2048}
    current = clean_fold()
    before = copy.deepcopy(current)
    texts, audit = context_prompts(current, tokenizer, spec)
    context = json.loads(tokenizer.messages[1]["content"])
    assert context["comment"] == current["queries"][0]["body"]
    assert len(context["violating_examples"]) == len(context["permitted_examples"]) == 1
    assert current == before
    assert len(texts) == 1 and audit["max_tokens"] < 2048
    with pytest.raises(ValueError, match="context budget"):
        context_prompts(current, tokenizer, {**spec, "max_tokens": 5})
