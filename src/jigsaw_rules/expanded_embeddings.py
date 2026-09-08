"""Extend a frozen embedding cache by input hash, preserving completed vectors."""

from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from filelock import FileLock

from jigsaw_rules.embeddings import QwenEncoder, content_key, load_spec, prepare_model
from jigsaw_rules.research import read_embedding_cache
from jigsaw_rules.runtime import Progress, atomic_json, stage
from jigsaw_rules.semantic import input_texts


def verified_keys(cache: Path, contract: dict) -> set[str]:
    """Discover keys only after verifying every committed shard and its schema."""
    if not cache.exists():
        return set()
    keys = []
    for path in sorted(cache.glob("batch_*/inputs.json")):
        keys.extend(json.loads(path.read_text()))
    if keys:
        read_embedding_cache(cache, contract, sorted(set(keys)))
    elif (cache / "contract.json").exists():
        if json.loads((cache / "contract.json").read_text()) != contract:
            raise ValueError("Incompatible embedding cache contract")
    return set(keys)


def prepare_plan(root: Path, texts: list[str], contract: dict, shard_size: int) -> Path:
    if not texts or shard_size < 1 or any(not isinstance(t, str) or not t for t in texts):
        raise ValueError("Nonempty texts and positive shard size required")
    cache = root / "runs/embeddings" / content_key(contract)[:20]
    existing = verified_keys(cache, contract)
    atomic_json(cache / "contract.json", contract)
    unique = sorted(set(texts))
    identity = {
        "encoder": contract,
        "requested_inputs": [content_key(t) for t in unique],
        "shard_size": shard_size,
    }
    directory = root / "runs/expanded_embeddings" / content_key(identity)[:20]

    def write(path):
        missing = [t for t in unique if content_key(t) not in existing]
        atomic_json(path / "identity.json", identity)
        atomic_json(
            path / "batches.json",
            [missing[i : i + shard_size] for i in range(0, len(missing), shard_size)],
        )
        atomic_json(
            path / "audit.json",
            {
                "occurrences": len(texts),
                "unique_inputs": len(unique),
                "already_cached": len(unique) - len(missing),
                "missing_at_start": len(missing),
            },
        )

    return stage(directory, "plan", write)


def encode_batches(root: Path, plan: Path, batches: list[list[str]], encoder, worker: int) -> dict:
    """Each process owns disjoint shards; only the parent writes the cache contract."""
    identity = json.loads((plan / "identity.json").read_text())
    if identity["encoder"] != encoder.contract:
        raise ValueError("Encoder differs from the committed work plan")
    cache = root / "runs/embeddings" / content_key(encoder.contract)[:20]
    encoded = reused = 0
    with Progress(plan.parent / f"worker_{worker}.jsonl", "expanded_embeddings") as log:
        for batch in batches:
            executed = False

            def encode(path, batch=batch):
                nonlocal executed
                executed = True
                array, statistics = encoder.encode(batch, log)
                if (
                    array.ndim != 2
                    or len(array) != len(batch)
                    or not np.isfinite(array).all()
                    or not np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-4)
                ):
                    raise ValueError("Invalid frozen embedding vectors")
                np.save(path / "vectors.npy", array.astype(np.float32), allow_pickle=False)
                atomic_json(path / "inputs.json", [content_key(t) for t in batch])
                atomic_json(path / "statistics.json", statistics)

            stage(cache, "batch_" + content_key(batch)[:20], encode)
            encoded += len(batch) if executed else 0
            reused += 0 if executed else len(batch)
            log.emit("shard_ready", worker=worker, encoded=encoded, reused=reused)
    return {"worker": worker, "encoded": encoded, "reused": reused}


def _worker(root: Path, plan: Path, batches: list[list[str]], spec: dict, worker: int) -> dict:
    return encode_batches(root, plan, batches, QwenEncoder(root, spec), worker)


def extend_embeddings(root: Path, frame, *, workers: int = 4, shard_size: int = 64) -> dict:
    if workers < 1:
        raise ValueError("At least one embedding worker required")
    spec = load_spec(root)
    contract = QwenEncoder(root, spec).contract
    texts = input_texts(frame, spec)
    (root / "runs").mkdir(exist_ok=True)
    with FileLock(str(root / "runs/expanded_embeddings.lock"), timeout=1):
        plan = prepare_plan(root, texts, contract, shard_size)
        batches = json.loads((plan / "batches.json").read_text())
        cache = root / "runs/embeddings" / content_key(contract)[:20]
        existing = verified_keys(cache, contract)
        pending = [b for b in batches if any(content_key(t) not in existing for t in b)]
        results = []
        if pending:
            # Avoid concurrent downloads of the same model files.
            prepare_model(root, spec)
            jobs = [pending[i::workers] for i in range(workers)]
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                futures = [
                    pool.submit(_worker, root, plan, batch, spec, i)
                    for i, batch in enumerate(jobs)
                    if batch
                ]
                results = [future.result() for future in futures]
        _, cache_audit = read_embedding_cache(cache, contract, [content_key(t) for t in texts])
        report = {
            **json.loads((plan / "audit.json").read_text()),
            "plan": plan.parent.name,
            "workers": results,
            "cache": cache_audit,
        }
        atomic_json(plan.parent / "completed.json", report)
        return report
