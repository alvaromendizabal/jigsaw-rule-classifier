"""Bounded GPU feature extraction with immutable input and durable shard checks."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from jigsaw_rules.data import validate_frame
from jigsaw_rules.runtime import Progress, atomic_json, digest
from scripts.competition_features import (
    PROMPTS,
    FrozenEncoder,
    cached_batch,
    content_hash,
    prepare_prompts,
)


def verified_model(spec, model_path):
    import hashlib

    from huggingface_hub import snapshot_download

    snapshot_download(
        spec["model_id"],
        revision=spec["revision"],
        local_dir=model_path,
        allow_patterns=list(spec["files"]),
    )
    for filename, expected in spec["files"].items():
        path = model_path / filename
        if expected["algorithm"] == "sha256":
            actual = digest(path)
        else:
            payload = path.read_bytes()
            actual = hashlib.sha1(
                b"blob " + str(len(payload)).encode() + b"\0" + payload
            ).hexdigest()
        if actual != expected["digest"]:
            raise ValueError("Pinned model asset checksum mismatch: " + filename)
    return model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    import boto3
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = json.loads(args.config.read_text())
    if digest(args.input) != spec["input_sha256"]:
        raise ValueError("Target-free development input differs from frozen manifest")
    frame = pd.read_csv(args.input)
    validate_frame(frame, train=False)
    if not torch.cuda.is_available():
        raise RuntimeError("This bounded experiment requires its declared CUDA GPU")
    torch.manual_seed(spec["seed"])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    contract = {
        "spec": spec,
        "source": {
            p.name: digest(p)
            for p in (Path(__file__), Path(__file__).with_name("competition_features.py"))
        },
        "software": {
            name: version(name) for name in ("torch", "transformers", "numpy", "huggingface-hub")
        },
        "python": platform.python_version(),
        "device": torch.cuda.get_device_name(),
    }
    run_id = content_hash(contract)[:20]
    cache = args.output / run_id
    s3 = boto3.client("s3", region_name="us-west-2")
    remote = args.prefix.rstrip("/") + "/checkpoints/" + run_id + "/"

    def upload(path):
        key = remote + path.relative_to(cache).as_posix()
        s3.upload_file(str(path), args.bucket, key, ExtraArgs={"ServerSideEncryption": "AES256"})

    # Restore only this exact source/config/software contract, not a nearby run.
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=args.bucket, Prefix=remote):
        for obj in page.get("Contents", []):
            relative = obj["Key"][len(remote) :]
            path = cache / relative
            if cache.resolve() not in path.resolve().parents:
                raise ValueError("Unsafe checkpoint path")
            path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(args.bucket, obj["Key"], str(path))
    cache.mkdir(parents=True, exist_ok=True)
    atomic_json(cache / "contract.json", contract)
    upload(cache / "contract.json")
    with Progress(cache / "events.jsonl", "competition_features", total_started=started) as log:
        log.emit("input_verified", rows=len(frame), run_id=run_id, device=contract["device"])
        model_path = verified_model(spec, args.output / "model")
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        texts, preparation = prepare_prompts(frame, tokenizer, spec)
        log.emit("prompts_ready", prompts=len(texts), **preparation)
        encoder = None

        def encode(batch):
            nonlocal encoder
            if encoder is None:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                    device_map="cuda",
                    local_files_only=True,
                    trust_remote_code=False,
                )
                encoder = FrozenEncoder(model, tokenizer)
                log.emit("model_loaded", gpu_memory_gib=torch.cuda.memory_allocated() / 2**30)
            pieces = [
                encoder.encode(batch[i : i + spec["batch_size"]])
                for i in range(0, len(batch), spec["batch_size"])
            ]
            return tuple(np.concatenate([p[j] for p in pieces]) for j in (0, 1))

        order = sorted(range(len(texts)), key=lambda i: (len(texts[i]), i))
        all_scores, all_vectors = {}, {}
        reused = 0
        for start in range(0, len(order), spec["shard_size"]):
            if time.monotonic() - started > spec["max_seconds"]:
                raise TimeoutError(
                    "Bounded feature runtime exhausted; completed shards are durable"
                )
            indices = order[start : start + spec["shard_size"]]
            batch = [texts[i] for i in indices]
            shard_key = content_hash({"contract": contract, "texts": batch})
            reused += int((cache / "shards" / shard_key / "complete.json").exists())
            scores, vectors, path = cached_batch(cache / "shards", contract, batch, encode)
            # Upload payload first and marker last, preserving transactional reuse.
            upload(path / "features.npz")
            upload(path / "complete.json")
            all_scores.update(zip(indices, scores, strict=True))
            all_vectors.update(zip(indices, vectors, strict=True))
            log.emit("shard_complete", prompts_done=len(all_scores), total_prompts=len(texts))
        scores = np.stack([all_scores[i] for i in range(len(texts))])
        vectors = np.stack([all_vectors[i] for i in range(len(texts))])
        output = cache / "representations.npz"
        np.savez_compressed(
            output,
            row_ids=frame.row_id.to_numpy(),
            scores=scores.reshape(len(frame), len(PROMPTS), -1),
            vectors=vectors.reshape(len(frame), len(PROMPTS), -1),
        )
        upload(output)
        record = {
            "run_id": run_id,
            "status": "completed",
            "rows": len(frame),
            "prompts": len(texts),
            "dimension": vectors.shape[-1],
            "reused_shards": reused,
            "input_sha256": digest(args.input),
            "representations_sha256": digest(output),
            "total_seconds": time.monotonic() - started,
            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
            **preparation,
        }
        atomic_json(cache / "complete.json", record)
        upload(cache / "complete.json")
        log.emit("completed", **record)
        upload(cache / "events.jsonl")
        s3.put_object(
            Bucket=args.bucket,
            Key=args.prefix.rstrip("/") + "/complete.json",
            Body=json.dumps({**record, "checkpoint_prefix": remote}).encode(),
            ServerSideEncryption="AES256",
        )
    # Model assets remain in the immutable Hub revision; do not package 8GB twice.
    Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model")).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
