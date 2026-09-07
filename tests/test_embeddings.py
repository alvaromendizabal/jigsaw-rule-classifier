import json

import numpy as np
import pytest
import torch

from jigsaw_rules.embeddings import encode_cached, last_token_pool
from tests.helpers import TestEncoder


def test_shards_resume_after_interruption_and_preserve_row_order(tmp_path):
    texts = ["third", "first", "second", "first", "fourth"]
    interrupted = TestEncoder(fail_on_call=2)
    with pytest.raises(RuntimeError, match="interruption"):
        encode_cached(tmp_path, texts, interrupted, shard_size=2)
    resumed = TestEncoder()
    vectors, stats = encode_cached(tmp_path, texts, resumed, shard_size=2)
    assert resumed.calls == 1
    assert stats["reused_texts"] == 2
    np.testing.assert_array_equal(vectors[1], vectors[3])
    full, _ = encode_cached(tmp_path / "uninterrupted", texts, TestEncoder(), shard_size=2)
    np.testing.assert_array_equal(vectors, full)


def test_corruption_recomputes_only_affected_shard(tmp_path):
    encoder = TestEncoder()
    expected, _ = encode_cached(tmp_path, ["one", "two", "three", "four"], encoder, shard_size=2)
    path = next((tmp_path / "runs/embeddings").rglob("vectors.npy"))
    path.write_bytes(b"interrupted output")
    restored, _ = encode_cached(tmp_path, ["one", "two", "three", "four"], encoder, shard_size=2)
    assert encoder.calls == 3
    np.testing.assert_array_equal(restored, expected)


def test_contract_change_invalidates_embeddings(tmp_path):
    encoder = TestEncoder()
    encode_cached(tmp_path, ["one", "two"], encoder)
    encoder.contract["prompt"] = "different instruction"
    encode_cached(tmp_path, ["one", "two"], encoder)
    assert encoder.calls == 2


def test_invalid_encoder_outputs_never_commit(tmp_path):
    encoder = TestEncoder()
    encoder.encode = lambda texts, log: (np.zeros((len(texts), 8)), {})
    with pytest.raises(ValueError, match="unit L2"):
        encode_cached(tmp_path, ["one"], encoder)
    assert not list((tmp_path / "runs/embeddings").rglob("complete.json"))


def test_pooling_handles_both_padding_directions():
    hidden = torch.arange(24).reshape(2, 4, 3).float()
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]])
    result = last_token_pool(hidden, mask)
    torch.testing.assert_close(result, torch.stack([hidden[0, 3], hidden[1, 1]]))


def test_pooling_rejects_empty_sequence():
    with pytest.raises(ValueError, match="nonempty"):
        last_token_pool(torch.ones(1, 3, 4), torch.zeros(1, 3))


def test_cache_contract_is_recorded(tmp_path):
    encoder = TestEncoder()
    encode_cached(tmp_path, ["one"], encoder)
    path = next((tmp_path / "runs/embeddings").glob("*/contract.json"))
    assert json.loads(path.read_text()) == encoder.contract
