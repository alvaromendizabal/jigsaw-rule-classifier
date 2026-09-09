"""Bounded cloud comparison of frozen and support-adapted decision representations."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import random
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from jigsaw_rules.runtime import Progress, atomic_json, digest
from scripts.competition_features import FrozenEncoder, content_hash
from scripts.support_adaptation import add_adapter, decision_prompts, train_adapter, validate_study


class ResumeProbe(Exception):
    """Controlled interruption after a durable optimizer checkpoint."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    import boto3
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = Path(__file__).resolve().parents[1]
    config = json.loads(args.config.read_text())
    model_spec = json.loads((root / "configs/competition_features.json").read_text())
    if digest(args.plan) != config["plan_sha256"]:
        raise ValueError("Adaptation plan differs from frozen input manifest")
    plan = json.loads(args.plan.read_text())
    validate_study(plan)
    frozen_record = json.loads(args.frozen.with_name("complete.json").read_text())
    if digest(args.frozen) != frozen_record["representations_sha256"]:
        raise ValueError("Frozen representation checksum differs")
    if digest(root / "input.csv") != model_spec["input_sha256"]:
        raise ValueError("Frozen input checksum differs")
    contract = {
        "config": config,
        "model_spec": model_spec,
        "frozen_sha256": digest(args.frozen),
        "source": {
            name: digest(root / "scripts" / name)
            for name in ("adaptation_worker.py", "support_adaptation.py", "competition_features.py")
        },
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
    s3 = boto3.client("s3", region_name="us-west-2")
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=args.bucket, Prefix=remote):
        for item in page.get("Contents", []):
            path = cache / item["Key"][len(remote) :]
            if cache.resolve() not in path.resolve().parents:
                raise ValueError("Unsafe checkpoint path")
            path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(args.bucket, item["Key"], str(path))

    def sync(path):
        s3.upload_file(
            str(path),
            args.bucket,
            remote + path.relative_to(cache).as_posix(),
            ExtraArgs={"ServerSideEncryption": "AES256"},
        )

    atomic_json(cache / "contract.json", contract)
    sync(cache / "contract.json")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, padding_side="left", local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    inputs = pd.read_csv(root / "input.csv")
    known_prompts = decision_prompts(inputs.to_dict("records"), tokenizer, model_spec)
    with np.load(args.frozen, allow_pickle=False) as data:
        if not np.array_equal(inputs.row_id.to_numpy(), data["row_ids"]):
            raise ValueError("Frozen cache IDs differ")
        known = {
            text: (score, vector)
            for text, score, vector in zip(
                known_prompts, data["scores"][:, 0], data["vectors"][:, 0], strict=True
            )
        }
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    records = []
    with Progress(cache / "events.jsonl", "support_adaptation") as log:
        for index, fold in enumerate(plan["folds"]):
            directory = cache / f"fold_{index}"
            marker = directory / "complete.json"
            if marker.exists():
                record = json.loads(marker.read_text())
                if digest(directory / "representations.npz") != record["sha256"]:
                    raise ValueError("Completed fold checksum differs")
                records.append(record)
                log.emit("fold_reused", fold=index)
                continue
            spec = {**config["training"], "run_contract": run_id, "fold": index}
            texts = decision_prompts(fold["training"], tokenizer, model_spec)
            queries = decision_prompts(fold["queries"], tokenizer, model_spec)
            base = None

            def load_base(seed=spec["seed"]):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                return AutoModelForCausalLM.from_pretrained(
                    args.model,
                    dtype=torch.bfloat16,
                    device_map="cuda",
                    attn_implementation="sdpa",
                    local_files_only=True,
                    trust_remote_code=False,
                )

            missing = sorted(set(texts + queries) - known.keys())
            if missing:
                base = load_base()
                encoder = FrozenEncoder(base, tokenizer)
                for start in range(0, len(missing), model_spec["batch_size"]):
                    batch = missing[start : start + model_spec["batch_size"]]
                    scores, vectors = encoder.encode(batch)
                    known.update(zip(batch, zip(scores, vectors, strict=True), strict=True))
                del encoder
            frozen = {
                prefix + "_" + name: np.stack([known[text][i] for text in sequence])
                for prefix, sequence in (("train_frozen", texts), ("query_frozen", queries))
                for i, name in enumerate(("scores", "vectors"))
            }
            log.emit(
                "frozen_features_ready",
                fold=index,
                reused=len(texts) + len(queries) - len(missing),
                encoded_missing=len(missing),
                training_pairs=len(texts),
                queries=len(queries),
            )
            if base is None:
                base = load_base()
            model = add_adapter(base, spec)
            del base
            probe = {"pending": not list((directory / "training").glob("step_*/complete.json"))}

            def checkpoint(path, probe=probe):
                sync(path)
                if time.monotonic() - started > config["max_seconds"]:
                    raise TimeoutError(
                        "Bounded adaptation runtime exhausted; optimizer state is durable"
                    )
                if probe["pending"] and path.name == "complete.json":
                    raise ResumeProbe()

            args_train = {
                "tokenizer": tokenizer,
                "texts": texts,
                "labels": [row["rule_violation"] for row in fold["training"]],
                "repeats": [row["repeat"] for row in fold["training"]],
                "spec": spec,
                "output": directory / "training",
                "log": log,
                "sync": checkpoint,
            }
            restart = False
            try:
                training = train_adapter(model=model, **args_train)
            except ResumeProbe:
                log.emit("resume_probe_restart", fold=index)
                restart = True
            if restart:
                del model
                gc.collect()
                torch.cuda.empty_cache()
                model = add_adapter(load_base(), spec)
                probe["pending"] = False
                training = train_adapter(model=model, **args_train)
                if training["resumed_step"] <= 0:
                    raise ValueError("Optimizer recovery did not resume")
            encoder = FrozenEncoder(model.get_base_model(), tokenizer)
            adapted = {}
            for prefix, sequence in (("train_adapted", texts), ("query_adapted", queries)):
                values = [
                    encoder.encode(sequence[start : start + model_spec["batch_size"]])
                    for start in range(0, len(sequence), model_spec["batch_size"])
                ]
                for i, name in enumerate(("scores", "vectors")):
                    adapted[prefix + "_" + name] = np.concatenate([value[i] for value in values])
                log.emit("adapted_features_ready", fold=index, cohort=prefix, rows=len(sequence))
            directory.mkdir(parents=True, exist_ok=True)
            output = directory / "representations.npz"
            np.savez_compressed(
                output,
                query_row_ids=[row["row_id"] for row in fold["queries"]],
                **frozen,
                **adapted,
            )
            sync(output)
            record = {
                "fold": index,
                "rule": fold["rule"],
                "audit": fold["audit"],
                "sha256": digest(output),
                "training": training,
                "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            atomic_json(marker, record)
            sync(marker)
            records.append(record)
            del encoder, model
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


if __name__ == "__main__":
    main()
