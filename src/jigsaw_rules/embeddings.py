"""Pinned Qwen inference and atomic, checksummed embedding shards."""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import subprocess
import sys
import time
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Protocol

import numpy as np
import psutil

from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest, stage

MODEL_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
    "model.safetensors",
]


def peak_memory_gib() -> float:
    import resource

    divisor = 2**30 if sys.platform == "darwin" else 2**20
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def content_key(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_spec(root: Path) -> dict:
    spec = json.loads((root / "configs/semantic.json").read_text())
    if spec["model_id"] != "Qwen/Qwen3-Embedding-0.6B" or len(spec["revision"]) != 40:
        raise ValueError("This encoder requires a pinned Qwen3 embedding revision")
    if spec["device"] != "cpu" or spec["dtype"] != "float32":
        raise ValueError("This release validates CPU float32 inference")
    for name in ["micro_batch", "cache_batch", "threads", "max_length", "max_seconds"]:
        if type(spec[name]) is not int or spec[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if spec["margin_temperature"] <= 0 or spec["classifier_c"] <= 0:
        raise ValueError("Temperature and regularization must be positive")
    return spec


def prepare_model(root: Path, spec: dict) -> Path:
    """Hub downloads resume files; verified local assets require no network."""
    destination = root / "models/qwen3-embedding-0.6b" / spec["revision"]
    record_path = destination / "manifest.json"
    prior = json.loads(record_path.read_text()) if record_path.exists() else {}
    official = json.loads((root / "configs/model.json").read_text())
    if official["model_id"] != spec["model_id"] or official["revision"] != spec["revision"]:
        raise ValueError("Model revision does not match the checked-in asset manifest")

    def verify_asset(path, name):
        entry = official["files"][name]
        if entry["algorithm"] == "sha256":
            actual = digest(path)
        else:
            payload = path.read_bytes()
            actual = hashlib.sha1(
                b"blob " + str(len(payload)).encode() + b"\0" + payload
            ).hexdigest()
        if actual != entry["digest"]:
            raise ValueError(f"Model asset differs from pinned Hub object: {name}")

    with Progress(root / "runs/model_download.jsonl", "model_download") as log:
        for name in MODEL_FILES:
            path = destination / name
            if path.is_file() and prior.get(name) == digest(path):
                verify_asset(path, name)
                log.emit("model_file_reused", file=name)
                continue
            command = [
                str(Path(sys.executable).with_name("hf")),
                "download",
                spec["model_id"],
                name,
                "--revision",
                spec["revision"],
                "--local-dir",
                str(destination),
                "--quiet",
            ]
            if path.exists():
                command.append("--force-download")
            subprocess.run(command, check=True, timeout=1200)
            verify_asset(path, name)
            prior[name] = digest(path)
            atomic_json(record_path, prior)
            log.emit("model_file_verified", file=name, bytes=path.stat().st_size)
    return destination


def format_text(rule: str, comment: str, spec: dict) -> str:
    context = f"Rule: {rule}\n" if spec["include_rule"] else ""
    return f"Instruct: {spec['instruction']}\nQuery: {context}Comment: {comment}"


def last_token_pool(hidden, mask):
    """Select the last non-padding position for either left or right padding."""
    import torch

    if hidden.ndim != 3 or mask.shape != hidden.shape[:2] or not (mask.sum(1) > 0).all():
        raise ValueError("Pooling requires aligned nonempty token sequences")
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    last = positions.masked_fill(mask == 0, -1).max(dim=1).values
    return hidden[torch.arange(len(hidden), device=hidden.device), last]


class Encoder(Protocol):
    contract: dict

    def encode(self, texts: list[str], log: Progress) -> tuple[np.ndarray, dict]: ...


class QwenEncoder:
    """Lazy model loading lets completely cached runs resume without loading weights."""

    def __init__(self, root: Path, spec: dict):
        self.root, self.spec = root, spec
        self.model = self.tokenizer = None
        self.deadline = time.monotonic() + spec["max_seconds"]
        self.contract = {
            **{
                k: spec[k]
                for k in [
                    "model_id",
                    "revision",
                    "max_length",
                    "micro_batch",
                    "threads",
                    "dtype",
                    "device",
                    "include_rule",
                    "instruction",
                ]
            },
            "pooling": "last_non_padding_token_l2_normalized",
            "software": {name: version(name) for name in ["torch", "transformers", "numpy"]},
            "implementation": content_key(
                [
                    inspect.getsource(type(self)),
                    inspect.getsource(last_token_pool),
                    inspect.getsource(format_text),
                ]
            ),
        }

    def _load(self, log: Progress) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        available = psutil.virtual_memory().available / 2**30
        if available < 4:
            raise MemoryError("Use a CPU workspace with at least 8 GB RAM and 4 GB currently free")
        torch.set_num_threads(self.spec["threads"])
        torch.manual_seed(self.spec["seed"])
        torch.use_deterministic_algorithms(True)
        directory = prepare_model(self.root, self.spec)
        self.tokenizer = AutoTokenizer.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False, padding_side="left"
        )
        self.model = AutoModel.from_pretrained(
            directory,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=torch.float32,
            attn_implementation="sdpa",
        ).eval()
        log.emit("model_loaded", revision=self.spec["revision"], device="cpu", dtype="float32")

    def encode(self, texts: list[str], log: Progress) -> tuple[np.ndarray, dict]:
        import torch
        import torch.nn.functional as functional

        self._load(log)
        output, lengths = [], []
        peak_rss = peak_memory_gib()
        started = time.monotonic()
        for start in range(0, len(texts), self.spec["micro_batch"]):
            if time.monotonic() > self.deadline:
                raise TimeoutError("Time budget reached; rerun to reuse completed embedding shards")
            batch = texts[start : start + self.spec["micro_batch"]]
            encoded = self.tokenizer(batch, add_special_tokens=True, truncation=False)
            lengths.extend(len(ids) for ids in encoded["input_ids"])
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.spec["max_length"],
                return_tensors="pt",
            )
            with torch.inference_mode():
                hidden = self.model(**inputs, use_cache=False).last_hidden_state
                vectors = functional.normalize(
                    last_token_pool(hidden, inputs["attention_mask"]), p=2, dim=1
                )
            output.append(vectors.cpu().numpy().astype(np.float32))
            done = min(start + len(batch), len(texts))
            peak_rss = max(peak_rss, peak_memory_gib())
            log.emit(
                "microbatch_processed",
                completed=done,
                total=len(texts),
                rows_per_second=round(done / max(time.monotonic() - started, 1e-9), 3),
                peak_rss_gib=round(peak_memory_gib(), 3),
            )
        return np.concatenate(output), {
            "encode_seconds": time.monotonic() - started,
            "rows": len(texts),
            "truncated_rows": sum(n > self.spec["max_length"] for n in lengths),
            "maximum_original_tokens": max(lengths),
            "peak_rss_gib": peak_rss,
        }


def encode_cached(
    root: Path,
    texts: list[str],
    encoder: Encoder,
    *,
    shard_size: int = 64,
    checkpoint: Callable[[], None] | None = None,
) -> tuple[np.ndarray, dict]:
    if not texts or shard_size < 1 or any(not isinstance(t, str) or not t for t in texts):
        raise ValueError("Nonempty texts and positive shard size are required")
    unique = sorted(set(texts))
    cache = root / "runs/embeddings" / content_key(encoder.contract)[:20]
    atomic_json(cache / "contract.json", encoder.contract)
    vectors, totals = (
        {},
        {
            "unique_texts": len(unique),
            "encoded_texts": 0,
            "reused_texts": 0,
            "truncated_rows": 0,
            "encode_seconds": 0.0,
        },
    )
    with Progress(root / "runs/embeddings.jsonl", "embedding_cache") as log:
        for start in range(0, len(unique), shard_size):
            batch = unique[start : start + shard_size]
            name = "batch_" + content_key(batch)[:20]
            executed = False

            def encode(destination, batch=batch):
                nonlocal executed
                executed = True
                array, stats = encoder.encode(batch, log)
                if array.ndim != 2 or len(array) != len(batch) or not np.isfinite(array).all():
                    raise ValueError("Encoder produced invalid vectors")
                if not np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-4):
                    raise ValueError("Encoder vectors must have unit L2 norm")
                stream = io.BytesIO()
                np.save(stream, array.astype(np.float32), allow_pickle=False)
                atomic_bytes(destination / "vectors.npy", stream.getvalue())
                atomic_json(destination / "statistics.json", stats)
                atomic_json(destination / "inputs.json", [content_key(t) for t in batch])

            directory = stage(cache, name, encode)
            array = np.load(directory / "vectors.npy", allow_pickle=False)
            stats = json.loads((directory / "statistics.json").read_text())
            totals["encoded_texts" if executed else "reused_texts"] += len(batch)
            totals["truncated_rows"] += stats["truncated_rows"]
            totals["encode_seconds"] += stats["encode_seconds"]
            totals["peak_rss_gib"] = max(
                totals.get("peak_rss_gib", 0), stats.get("peak_rss_gib", 0)
            )
            vectors.update(zip(batch, array, strict=True))
            if checkpoint is not None and executed:
                checkpoint()
            log.emit(
                "embedding_shard_ready",
                completed=min(start + len(batch), len(unique)),
                total=len(unique),
                reused=not executed,
            )
    return np.stack([vectors[t] for t in texts]), totals
