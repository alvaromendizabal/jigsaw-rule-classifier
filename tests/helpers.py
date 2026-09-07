"""Deterministic test doubles; never report their metrics as model performance."""

import hashlib

import numpy as np


class TestEncoder:
    __test__ = False

    def __init__(self, *, fail_on_call=None):
        self.contract = {"software_test": True, "backend": "sha256_test_vectors"}
        self.calls = 0
        self.fail_on_call = fail_on_call

    def encode(self, texts, log):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("Simulated interruption")
        values = np.array(
            [list(hashlib.sha256(t.encode()).digest()) for t in texts], dtype=np.float32
        )
        values -= 127.5
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        return values, {"rows": len(texts), "encode_seconds": 0.0, "truncated_rows": 0}
