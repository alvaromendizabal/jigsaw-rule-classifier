"""Bounded real-model integration: pinned assets, padding invariance, and cache reuse."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from jigsaw_rules.embeddings import (
    QwenEncoder,
    encode_cached,
    format_text,
    load_spec,
    peak_memory_gib,
)
from jigsaw_rules.runtime import Progress, atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    spec = load_spec(root)
    encoder = QwenEncoder(root, spec)
    texts = [
        format_text("No advertising", comment, spec)
        for comment in [
            "Buy our new product today.",
            "I enjoyed reading the article.",
            "A discount offer is available on our website.",
            "Thank you for the useful explanation.",
        ]
    ]
    started = time.monotonic()
    with Progress(root / "logs/semantic_integration.jsonl", "semantic_integration") as log:
        batched, _ = encoder.encode(texts, log)
        individual = np.concatenate([encoder.encode([text], log)[0] for text in texts])
        np.testing.assert_allclose(batched, individual, atol=2e-5, rtol=2e-4)
        np.testing.assert_allclose(np.linalg.norm(batched, axis=1), 1, atol=1e-5)
        if np.allclose(batched[0], batched[1]):
            raise AssertionError("The encoder returned constant vectors")
        cached, _ = encode_cached(root, texts, encoder, shard_size=2)

        def forbidden(*args, **kwargs):
            raise AssertionError("Completed embeddings must not be encoded again")

        encoder.encode = forbidden
        resumed, stats = encode_cached(root, texts, encoder, shard_size=2)
        np.testing.assert_array_equal(cached, resumed)
        record = {
            "status": "passed",
            "data_kind": "authored integration examples",
            "model_id": spec["model_id"],
            "revision": spec["revision"],
            "contract": encoder.contract,
            "rows": len(texts),
            "dimensions": batched.shape[1],
            "maximum_batch_difference": float(np.max(np.abs(batched - individual))),
            "reused_rows": stats["reused_texts"],
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_gib": peak_memory_gib(),
            "note": "Real model integration, not a competition performance result.",
        }
        atomic_json(root / "reports/semantic/integration.json", record)
        log.emit("SEMANTIC_INTEGRATION_PASSED", rows=len(texts), dimensions=batched.shape[1])


if __name__ == "__main__":
    main()
