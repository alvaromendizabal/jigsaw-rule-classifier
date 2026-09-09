"""Offline support learning and rank-preserving competition inference."""

from __future__ import annotations

import hashlib
import io
import json
import platform
import random
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from jigsaw_rules.data import load_data, normalize, validate_submission
from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest
from scripts.competition_features import FrozenEncoder, content_hash, support_pairs
from scripts.decision_training import train_adapter
from scripts.support_adaptation import add_adapter, decision_prompts


def prepare_model(mirror: Path, destination: Path, spec: dict, assets: Path) -> Path:
    """Use matching weights and restore the exact upstream tokenizer configuration."""
    destination.mkdir(parents=True, exist_ok=True)
    for name, record in spec["files"].items():
        if Path(name).name != name:
            raise ValueError("Unsafe model asset name")
        path = destination / name
        if name in {"LICENSE", "tokenizer_config.json"}:
            atomic_bytes(path, (assets / name).read_bytes())
        elif not path.exists():
            path.symlink_to((mirror / name).resolve())
        algorithm = record["algorithm"]
        if algorithm not in {"sha256", "git-sha1"}:
            raise ValueError("Unsupported model checksum algorithm")
        checksum = hashlib.sha256() if algorithm == "sha256" else hashlib.sha1()
        if algorithm == "git-sha1":
            checksum.update(f"blob {path.stat().st_size}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(block)
        if checksum.hexdigest() != record["digest"]:
            raise ValueError(f"Pinned model checksum differs: {name}")
        print("MODEL_ASSET_VERIFIED", name, flush=True)
    return destination


def adaptation_rows(training, available):
    pairs, audit = support_pairs(training, available)
    familiar = set(training.rule.map(normalize))
    repeats = [1 if normalize(rule) in familiar else 2 for rule in pairs.rule]
    return (
        pairs,
        repeats,
        {
            **audit,
            "epoch_occurrences": sum(repeats),
            "new_policies": len(set(available.rule.map(normalize)) - familiar),
            "query_body_targets_used": False,
            "supplied_support_matches_are_eligible": True,
        },
    )


def policy_ranks(margins, rules):
    margins = np.asarray(margins, dtype=np.float64)
    if len(margins) != len(rules) or not np.isfinite(margins).all():
        raise ValueError("Invalid decision scores")
    result = np.empty(len(margins), dtype=np.float64)
    for rule in sorted(set(rules)):
        indices = np.flatnonzero(np.asarray(rules) == rule)
        result[indices] = (rankdata(margins[indices], method="average") - 0.5) / len(indices)
    return result


def cached_scores(path: Path, indices, texts, encoder):
    contract = content_hash({"indices": list(map(int, indices)), "texts": texts})
    marker = path / "complete.json"
    if marker.exists():
        record = json.loads(marker.read_text())
        if record["contract"] != contract or digest(path / "scores.npz") != record["sha256"]:
            raise ValueError("Decision shard contract/checksum differs")
    else:
        scores, _ = encoder.encode(texts)
        if scores.shape != (len(texts), 3) or not np.isfinite(scores).all():
            raise ValueError("Nonfinite or misaligned decision scores")
        payload = io.BytesIO()
        np.savez_compressed(payload, indices=indices, margins=scores[:, 1].astype(np.float64))
        atomic_bytes(path / "scores.npz", payload.getvalue())
        atomic_json(marker, {"contract": contract, "sha256": digest(path / "scores.npz")})
    with np.load(path / "scores.npz", allow_pickle=False) as data:
        if not np.array_equal(data["indices"], indices):
            raise ValueError("Decision shard row order differs")
        return data["margins"]


def run(input_root: Path, model_path: Path, output: Path, model_spec: dict, spec: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from jigsaw_rules import data as data_module
    from jigsaw_rules import runtime as runtime_module

    training, test, sample = load_data(input_root)
    if digest(input_root / "train.csv") != spec["training_sha256"]:
        raise ValueError("Only the original competition training file is accepted")
    if not torch.cuda.is_available():
        raise RuntimeError("This submission requires a Kaggle GPU accelerator")
    sources = {
        name: digest(Path(__file__).parent / name)
        for name in [
            "kaggle_adaptation.py",
            "decision_training.py",
            "support_adaptation.py",
            "competition_features.py",
        ]
    }
    sources.update(
        {
            "data.py": digest(Path(data_module.__file__)),
            "runtime.py": digest(Path(runtime_module.__file__)),
        }
    )
    contract = {
        "data": {
            name: digest(input_root / name)
            for name in ["train.csv", "test.csv", "sample_submission.csv"]
        },
        "model": model_spec,
        "settings": spec,
        "source": sources,
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ["torch", "transformers", "peft", "numpy", "pandas", "scipy"]
        },
        "device": torch.cuda.get_device_name(),
    }
    key = content_hash(contract)[:20]
    cache = output / "checkpoints" / key
    marker = cache / "complete.json"
    if marker.exists():
        record = json.loads(marker.read_text())
        if record["run_id"] != key or digest(cache / "submission.csv") != record["sha256"]:
            raise ValueError("Completed submission checksum differs")
        validate_submission(pd.read_csv(cache / "submission.csv"), sample)
        atomic_bytes(output / "submission.csv", (cache / "submission.csv").read_bytes())
        atomic_json(output / "submission_manifest.json", record)
        print("COMPLETED_SUBMISSION_REUSED", key, flush=True)
        return output / "submission.csv"
    atomic_json(cache / "contract.json", contract)
    started = time.monotonic()
    torch.set_num_threads(4)
    random.seed(spec["training"]["seed"])
    np.random.seed(spec["training"]["seed"])
    torch.manual_seed(spec["training"]["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, padding_side="left", local_files_only=True, trust_remote_code=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    pairs, repeats, audit = adaptation_rows(training, test)
    train_texts = decision_prompts(pairs.to_dict("records"), tokenizer, model_spec)
    query_texts = decision_prompts(test.to_dict("records"), tokenizer, model_spec)
    order = sorted(range(len(test)), key=lambda i: len(tokenizer.encode(query_texts[i])))
    precision = "bfloat16" if torch.cuda.is_bf16_supported(including_emulation=False) else "float16"
    training_spec = {**spec["training"], "precision": precision, "run_contract": key}
    with Progress(cache / "events.jsonl", "support_adapted_submission") as log:
        log.emit("input_audit", training_rows=len(training), query_rows=len(test), **audit)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=getattr(torch, precision),
            device_map="cuda:0",
            attn_implementation="sdpa",
            local_files_only=True,
            trust_remote_code=False,
        )
        model = add_adapter(model, training_spec)

        def time_guard(path=None):
            if time.monotonic() - started > spec["max_seconds"]:
                raise TimeoutError("Runtime budget reached; completed work remains resumable")

        fit = train_adapter(
            model,
            tokenizer,
            train_texts,
            pairs.rule_violation.tolist(),
            repeats,
            training_spec,
            cache / "training",
            log,
            sync=time_guard,
        )
        encoder = FrozenEncoder(model.get_base_model(), tokenizer, device="cuda:0")
        margins = np.empty(len(test), dtype=np.float64)
        for offset in range(0, len(order), spec["inference_batch"]):
            indices = order[offset : offset + spec["inference_batch"]]
            margins[indices] = cached_scores(
                cache / "predictions" / f"batch_{offset:08d}",
                indices,
                [query_texts[i] for i in indices],
                encoder,
            )
            time_guard()
            if offset % 128 == 0:
                log.emit(
                    "predictions_checkpointed",
                    rows=min(offset + len(indices), len(test)),
                    total=len(test),
                )
        submission = pd.DataFrame(
            {
                "row_id": test.row_id,
                "rule_violation": policy_ranks(margins, test.rule.to_numpy()),
            }
        )
        validate_submission(submission, sample)
        atomic_bytes(cache / "submission.csv", submission.to_csv(index=False).encode())
        record = {
            "run_id": key,
            "sha256": digest(cache / "submission.csv"),
            "rows": len(test),
            "training": fit,
            "audit": audit,
            "precision": precision,
            "elapsed_seconds": time.monotonic() - started,
            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
            "score_kind": "Within-policy ranks preserve AUC; not calibrated probabilities",
            "provenance": contract,
        }
        atomic_json(marker, record)
        atomic_bytes(output / "submission.csv", (cache / "submission.csv").read_bytes())
        atomic_json(output / "submission_manifest.json", record)
        log.emit("submission_ready", rows=len(test), run_id=key)
    return output / "submission.csv"
