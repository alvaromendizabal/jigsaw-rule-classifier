"""Bounded real-artifact parity, network isolation, serving budget and restore evidence."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import time
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import psutil

from jigsaw_rules.confirmation import read_inputs
from jigsaw_rules.data import EXAMPLES
from jigsaw_rules.embeddings import content_key, peak_memory_gib
from jigsaw_rules.offline import OfflineModel, package_model, predict_file
from jigsaw_rules.research import read_embedding_cache
from jigsaw_rules.runtime import Progress, atomic_json, digest
from jigsaw_rules.semantic import input_texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    spec = json.loads((root / "configs/delivery.json").read_text())
    work = root / "runs/delivery_verification"
    work.mkdir(parents=True, exist_ok=True)
    bundle = package_model(root, root / "runs/delivery")
    pin = digest(bundle / "bundle.json")
    private = root / "runs/confirmation/314494da886e11bcc1f6"
    frame, assignments = read_inputs(root, private)
    selected = assignments.drop_duplicates("rule_id").index.to_numpy()
    sample = frame.iloc[selected].reset_index(drop=True)
    frozen = pd.read_csv(private / "predictions/predictions.csv").iloc[selected]
    assert np.array_equal(frozen.row_id, sample.row_id)
    sample.to_csv(work / "inputs.csv", index=False)
    demo = pd.DataFrame(json.loads((root / "configs/demo.json").read_text()))
    os.environ.update(
        {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}
    )
    started = time.monotonic()
    original_open = Path.open

    def open_without_targets(path, *arguments, **keywords):
        if path.name == "solution.csv":
            raise AssertionError("Delivery verification must not reopen protected labels")
        return original_open(path, *arguments, **keywords)

    with (
        Progress(work / "events.jsonl", "offline_delivery") as log,
        patch.object(
            socket.socket, "connect", side_effect=AssertionError("Network forbidden")
        ) as network,
        patch.object(
            socket, "create_connection", side_effect=AssertionError("Network forbidden")
        ) as connect,
        patch.object(Path, "open", open_without_targets),
    ):
        cold_started = time.monotonic()
        model = OfflineModel(bundle, pin, work / "cache/encoding")
        cold_verified = time.monotonic() - cold_started
        predictions, details = model.predict(sample)
        cold_seconds = time.monotonic() - cold_started
        difference = float(np.max(np.abs(predictions.rule_violation.to_numpy() - frozen.candidate)))
        np.testing.assert_allclose(
            predictions.rule_violation, frozen.candidate, atol=spec["prediction_tolerance"], rtol=0
        )
        keys = [content_key(text) for text in input_texts(sample, model.spec)]
        cache = model.cache / "runs/embeddings" / content_key(model.encoder.contract)[:20]
        flat, _ = read_embedding_cache(cache, model.encoder.contract, keys)
        vectors = flat.reshape(len(sample), 5, -1)
        original = joblib.load(
            root / "runs/model_validation/a971cf3bc6add1c2d818/final/candidate.joblib"
        )
        vector_difference = float(
            np.max(np.abs(original.predict(sample, vectors) - predictions.rule_violation))
        )
        assert vector_difference <= spec["vector_parity_tolerance"]
        batches = np.concatenate(
            [
                model.predict_vectors(sample.iloc[i : i + 1], vectors[i : i + 1])
                for i in range(len(sample))
            ]
        )
        np.testing.assert_allclose(
            batches, predictions.rule_violation, atol=spec["vector_parity_tolerance"], rtol=0
        )
        reversed_scores, _ = model.predict(sample.iloc[::-1])
        np.testing.assert_allclose(
            reversed_scores.rule_violation.to_numpy()[::-1],
            predictions.rule_violation,
            atol=spec["vector_parity_tolerance"],
            rtol=0,
        )
        warm_started = time.monotonic()
        demo_scores, demo_details = model.predict(demo)
        warm_seconds = time.monotonic() - warm_started
        with patch.object(
            model.encoder, "encode", side_effect=AssertionError("Must reuse embeddings")
        ):
            replay, _ = model.predict(demo)
        np.testing.assert_array_equal(replay.rule_violation, demo_scores.rule_violation)
        file_args = (bundle, pin, work / "inputs.csv", work / "output", work / "cache")
        result = predict_file(*file_args, batch_rows=len(sample))
        first = result.read_bytes()
        with patch.object(
            OfflineModel, "predict", side_effect=AssertionError("Must reuse batches")
        ):
            assert predict_file(*file_args, batch_rows=len(sample)).read_bytes() == first
        assert network.call_count == connect.call_count == 0
        examples = []
        for i, row in demo.iterrows():
            column = demo_details["nearest_support_column"][i]
            examples.append(
                {
                    "name": row.row_id,
                    "comment": row.body,
                    "rule": row.rule,
                    "probability": float(demo_scores.rule_violation.iloc[i]),
                    "route": "familiar" if demo_details["familiar"][i] else "unseen",
                    "nearest_example": row[column],
                    "example_label": "violation" if column in EXAMPLES[:2] else "permitted",
                }
            )
        seconds, rss = time.monotonic() - started, peak_memory_gib()
        guards = {
            "frozen_prediction_parity": difference <= spec["prediction_tolerance"],
            "identical_vector_parity": vector_difference <= spec["vector_parity_tolerance"],
            "cold_start_budget": cold_seconds <= spec["maximum_cold_start_seconds"],
            "warm_budget": warm_seconds / len(demo) <= spec["maximum_warm_seconds_per_comment"],
            "memory_budget": rss <= spec["maximum_peak_rss_gib"],
            "total_budget": seconds <= spec["maximum_validation_seconds"],
            "network_calls_zero": network.call_count == connect.call_count == 0,
        }
        report = {
            "schema": 1,
            "status": "passed" if all(guards.values()) else "failed",
            "checks": guards,
            "bundle": str(bundle.relative_to(root)),
            "bundle_manifest_sha256": pin,
            "bundle_bytes": sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file()),
            "source_sha256": digest(Path(__file__)),
            "config_sha256": digest(root / "configs/delivery.json"),
            "demo_sha256": digest(root / "configs/demo.json"),
            "rows": len(sample),
            "policies": assignments.iloc[selected].rule_id.tolist(),
            "row_ids_sha256": content_key(sample.row_id.tolist()),
            "maximum_frozen_prediction_difference": difference,
            "maximum_identical_vector_difference": vector_difference,
            "cold_seconds_including_verification": cold_seconds,
            "bundle_verification_seconds": cold_verified,
            "warm_authored_batch_seconds": warm_seconds,
            "warm_authored_seconds_per_comment": warm_seconds / len(demo),
            "peak_rss_gib": rss,
            "total_seconds": seconds,
            "host": {
                "processor": platform.processor(),
                "machine": platform.machine(),
                "logical_cpus": psutil.cpu_count(),
                "threads_used": model.spec["threads"],
                "python": platform.python_version(),
            },
            "network_calls": 0,
            "targets_opened": False,
            "models_refitted": 0,
            "order_parity_passed": True,
            "batch_parity_passed": True,
            "embedding_reuse_passed": True,
            "prediction_reuse_passed": True,
            "examples": examples,
            "scope": (
                "Six target-free parity rows plus four authored examples; serving measurements "
                "on this host, not new performance estimates or a moderation threshold"
            ),
        }
        atomic_json(work / "verification.json", report)
        if not all(guards.values()):
            raise ValueError(f"Offline delivery guard failed: {guards}")
        atomic_json(root / "reports/delivery/verification.json", report)
        log.emit("OFFLINE_DELIVERY_VERIFIED", rows=len(sample), elapsed=seconds, peak_rss_gib=rss)
    print(json.dumps({"bundle": str(bundle), "manifest_sha256": pin, "status": report["status"]}))


if __name__ == "__main__":
    main()
