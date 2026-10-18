import json

import numpy as np
import pandas as pd

from jigsaw_rules.data import EXAMPLES


class Tokenizer:
    def encode(self, text, **kwargs):
        return text.split()

    def decode(self, tokens, **kwargs):
        return " ".join(tokens)

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return json.dumps(messages)


def frame(rows=30):
    result = pd.DataFrame(
        {
            "row_id": np.arange(rows),
            "body": [
                f"Can you help with this example number {i}? " + "word " * (i % 4)
                for i in range(rows)
            ],
            "rule": ["Do not offer advice."] * rows,
            "subreddit": [f"community_{i % 3}" for i in range(rows)],
            "rule_violation": np.arange(rows) % 2,
        }
    )
    for i, name in enumerate(EXAMPLES):
        result[name] = [
            f"Provided support {i}: " + "context " * (j % 3 + i + 1) for j in range(rows)
        ]
    return result
