"""Fixed Phi support study with durable shards and complete optimizer recovery."""

from __future__ import annotations

import argparse
import gc
import io
import json
import platform
import random
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest
from scripts.competition_features import FrozenEncoder, cached_batch, content_hash
from scripts.competition_worker import verified_model
from scripts.decision_training import train_adapter
from scripts.support_adaptation import add_adapter, decision_prompts, validate_study

SOURCE_FILES = (
    "scripts/complementary_worker.py",
    "scripts/competition_features.py",
    "scripts/competition_worker.py",
    "scripts/decision_training.py",
    "scripts/support_adaptation.py",
    "scripts/bootstrap_gpu.py",
    "scripts/run_complementarity.sh",
    "src/jigsaw_rules/data.py",
    "src/jigsaw_rules/runtime.py",
)


class ResumeProbe(Exception):
    """Intentional interruption at a complete optimizer boundary."""


def verified_plan(path, spec):
    if digest(path) != spec["plan_sha256"]:
        raise ValueError("Study plan checksum differs")
    plan = json.loads(path.read_text())
    validate_study(plan)
    return plan


def restore_keys(keys):
    """Restore artifacts and latest two committed states, never partial states."""
    markers = sorted(k for k in keys if "/training/step_" in k and k.endswith("/complete.json"))
    groups = {}
    for marker in markers:
        groups.setdefault(marker.split("/training/")[0], []).append(marker)
    states = {
        name
        for values in groups.values()
        for marker in values[-2:]
        for name in (marker, marker.removesuffix("complete.json") + "state.pt")
    }
    return sorted(k for k in keys if "/training/step_" not in k or k in states)


def checked_fold(directory, index, fold):
    record = json.loads((directory / "complete.json").read_text())
    if record["fold"] != index or record["rule"] != fold["rule"]:
        raise ValueError("Completed fold identity differs")
    artifact = directory / "representations.npz"
    if digest(artifact) != record["sha256"]:
        raise ValueError("Completed fold checksum differs")
    with np.load(artifact, allow_pickle=False) as data:
        if not np.array_equal(data["query_row_ids"], [r["row_id"] for r in fold["queries"]]):
            raise ValueError("Completed query order differs")
        for key in ("query_frozen_scores", "query_adapted_scores"):
            if data[key].shape != (len(fold["queries"]), 3) or not np.isfinite(data[key]).all():
                raise ValueError("Invalid completed scores")
    return record


def encode_queries(encoder, texts, directory, contract, spec, log, sync, check_time):
    order = sorted(range(len(texts)), key=lambda i: (len(texts[i]), i))
    scores, vectors = {}, {}
    for start in range(0, len(order), spec["shard_size"]):
        check_time()
        indices = order[start : start + spec["shard_size"]]
        batch = [texts[i] for i in indices]

        def encode(values):
            pieces = [
                encoder.encode(values[i : i + spec["batch_size"]])
                for i in range(0, len(values), spec["batch_size"])
            ]
            return tuple(np.concatenate([p[j] for p in pieces]) for j in (0, 1))

        a, b, path = cached_batch(directory, contract, batch, encode)
        sync(path / "features.npz")
        sync(path / "complete.json")
        scores.update(zip(indices, a, strict=True))
        vectors.update(zip(indices, b, strict=True))
        log.emit(
            "query_shard_complete", cohort=contract["cohort"], done=len(scores), total=len(texts)
        )
    return np.stack([scores[i] for i in range(len(texts))]), np.stack(
        [vectors[i] for i in range(len(texts))]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    import boto3
    import torch
    from botocore.config import Config
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/complementarity.json").read_text())
    model_spec = json.loads((root / "configs/complementary_model.json").read_text())
    plan = verified_plan(args.plan, config)
    if not torch.cuda.is_available():
        raise RuntimeError("The declared L4 GPU is required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    contract = {
        "config": config,
        "model_spec": model_spec,
        "source": {name: digest(root / name) for name in SOURCE_FILES},
        "software": {
            name: version(name)
            for name in ("torch", "transformers", "peft", "numpy", "huggingface-hub")
        },
        "python": platform.python_version(),
        "device": torch.cuda.get_device_name(),
    }
    run_id = content_hash(contract)[:20]
    cache = args.output / run_id
    remote = args.prefix.rstrip("/") + "/checkpoints/" + run_id + "/"
    s3 = boto3.client("s3", region_name="us-west-2", config=Config(retries={"mode": "standard"}))
    keys = [
        obj["Key"]
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=args.bucket, Prefix=remote)
        for obj in page.get("Contents", [])
    ]
    for key in restore_keys(keys):
        path = cache / key[len(remote) :]
        if cache.resolve() not in path.resolve().parents:
            raise ValueError("Unsafe remote checkpoint path")
        path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(args.bucket, key, str(path))

    def sync(path):
        s3.upload_file(
            str(path),
            args.bucket,
            remote + path.relative_to(cache).as_posix(),
            ExtraArgs={"ServerSideEncryption": "AES256"},
        )

    def check_time():
        if time.monotonic() - started > config["max_seconds"]:
            raise TimeoutError("Study budget exhausted; committed optimizer/shards are durable")

    atomic_json(cache / "contract.json", contract)
    sync(cache / "contract.json")
    with Progress(cache / "events.jsonl", "complementarity", total_started=started) as log:
        if (cache / "complete.json").exists():
            for i, fold in enumerate(plan["folds"]):
                checked_fold(cache / f"fold_{i}", i, fold)
            log.emit("study_reused_without_model", run_id=run_id)
            return
        log.emit(
            "plan_verified", run_id=run_id, queries=sum(len(f["queries"]) for f in plan["folds"])
        )
        model_path = verified_model(model_spec, args.output / "model")
        log.emit(
            "model_assets_verified", files=len(model_spec["files"]), revision=model_spec["revision"]
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, padding_side="left", local_files_only=True, trust_remote_code=False
        )
        tokenizer.pad_token = tokenizer.eos_token
        records = []
        for index, fold in enumerate(plan["folds"]):
            check_time()
            directory = cache / f"fold_{index}"
            if (directory / "complete.json").exists():
                records.append(checked_fold(directory, index, fold))
                log.emit("fold_reused", fold=index)
                continue
            spec = {**config["training"], "run_contract": run_id, "fold": index}
            texts = decision_prompts(fold["training"], tokenizer, model_spec)
            queries = decision_prompts(fold["queries"], tokenizer, model_spec)

            def load_model(spec=spec):
                random.seed(spec["seed"])
                np.random.seed(spec["seed"])
                torch.manual_seed(spec["seed"])
                return AutoModelForCausalLM.from_pretrained(
                    model_path,
                    dtype=torch.bfloat16,
                    device_map="cuda",
                    attn_implementation="sdpa",
                    local_files_only=True,
                    trust_remote_code=False,
                )

            base = load_model()
            frozen = encode_queries(
                FrozenEncoder(base, tokenizer),
                queries,
                directory / "frozen_shards",
                {"run_id": run_id, "fold": index, "cohort": "frozen"},
                model_spec,
                log,
                sync,
                check_time,
            )
            model = add_adapter(base, spec)
            del base
            log.emit("fold_training_start", fold=index, pairs=len(texts), queries=len(queries))
            probe = {"pending": not list((directory / "training").glob("step_*/complete.json"))}

            def checkpoint(path, probe=probe):
                sync(path)
                if path.name == "complete.json":
                    sync(cache / "events.jsonl")
                    check_time()
                    if probe["pending"]:
                        raise ResumeProbe("Controlled durable optimizer recovery probe")

            training_args = {
                "tokenizer": tokenizer,
                "texts": texts,
                "labels": [r["rule_violation"] for r in fold["training"]],
                "repeats": [r["repeat"] for r in fold["training"]],
                "spec": spec,
                "output": directory / "training",
                "log": log,
                "sync": checkpoint,
            }
            restart = False
            try:
                training = train_adapter(model=model, **training_args)
            except ResumeProbe:
                restart = True
            if restart:
                del model
                gc.collect()
                torch.cuda.empty_cache()
                probe["pending"] = False
                model = add_adapter(load_model(), spec)
                training = train_adapter(model=model, **training_args)
                if training["resumed_step"] <= 0:
                    raise ValueError("Optimizer probe did not resume") from None
                log.emit("optimizer_recovery_verified", fold=index, step=training["resumed_step"])
            adapted = encode_queries(
                FrozenEncoder(model.get_base_model(), tokenizer),
                queries,
                directory / "adapted_shards",
                {"run_id": run_id, "fold": index, "cohort": "adapted"},
                model_spec,
                log,
                sync,
                check_time,
            )
            payload = io.BytesIO()
            np.savez_compressed(
                payload,
                query_row_ids=[r["row_id"] for r in fold["queries"]],
                query_frozen_scores=frozen[0],
                query_frozen_vectors=frozen[1],
                query_adapted_scores=adapted[0],
                query_adapted_vectors=adapted[1],
            )
            artifact = directory / "representations.npz"
            atomic_bytes(artifact, payload.getvalue())
            sync(artifact)
            record = {
                "fold": index,
                "rule": fold["rule"],
                "audit": fold["audit"],
                "sha256": digest(artifact),
                "training": training,
                "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            atomic_json(directory / "complete.json", record)
            sync(directory / "complete.json")
            records.append(checked_fold(directory, index, fold))
            del model
            gc.collect()
            torch.cuda.empty_cache()
            log.emit("fold_complete", **record)
            sync(cache / "events.jsonl")
        result = {
            "status": "completed",
            "run_id": run_id,
            "folds": records,
            "elapsed_seconds": time.monotonic() - started,
            "checkpoint_prefix": remote,
        }
        atomic_json(cache / "complete.json", result)
        sync(cache / "complete.json")
        s3.put_object(
            Bucket=args.bucket,
            Key=args.prefix.rstrip("/") + "/complete.json",
            Body=json.dumps(result).encode(),
            ServerSideEncryption="AES256",
        )
        log.emit("study_complete", run_id=run_id)
    sync(cache / "events.jsonl")


if __name__ == "__main__":
    main()
