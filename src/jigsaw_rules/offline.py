"""Verified offline delivery of the accepted route, with resumable prediction batches."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from filelock import FileLock

from jigsaw_rules.confirmation import load_protocol
from jigsaw_rules.confirmation_report import confirmation_evidence
from jigsaw_rules.data import EXAMPLES, validate_frame, validate_submission
from jigsaw_rules.embeddings import MODEL_FILES, QwenEncoder, content_key, encode_cached, load_spec
from jigsaw_rules.final_model import familiar_mask, verify_stage
from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest, stage
from jigsaw_rules.semantic import input_texts

PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "torch", "transformers")


def software_versions() -> dict[str, str]:
    return {name: version(name) for name in PACKAGES}


def verify_bundle(bundle: Path, expected_sha256: str) -> dict:
    """Require an external manifest pin before reading trusted serialized model bytes."""
    bundle = bundle.resolve()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("A published bundle manifest SHA-256 is required")
    if digest(bundle / "bundle.json") != expected_sha256:
        raise ValueError("Bundle manifest differs from its published checksum")
    manifest = json.loads((bundle / "bundle.json").read_text())
    if manifest.get("schema") != 1 or manifest.get("kind") != "accepted_postcompetition_route":
        raise ValueError("Unsupported offline model bundle")
    if manifest["packages"] != software_versions():
        raise ValueError("Offline environment differs; install the bundle's locked dependencies")
    for relative, expected in manifest["files"].items():
        path = bundle / relative
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(bundle)
            or digest(path) != expected
        ):
            raise ValueError(f"Offline artifact checksum or path differs: {relative}")
    package = Path(__file__).parent
    sources = {p.name: digest(p) for p in sorted(package.glob("*.py"))}
    if sources != manifest["sources"]:
        raise ValueError("Installed inference source differs from the bundled source")
    if manifest["files"]["candidate.joblib"] != manifest["candidate_sha256"]:
        raise ValueError("Candidate identity differs from protected acceptance")
    return manifest


def package_model(root: Path, destination: Path) -> Path:
    """Package the exact accepted artifact and locally restored encoder without refitting."""
    evidence, plan, spec = confirmation_evidence(root), load_protocol(root), load_spec(root)
    if evidence is None or evidence["results"]["status"] != "accepted":
        raise ValueError("A verified accepted protected result is required before packaging")
    if evidence["metadata"]["candidate_sha256"] != plan["candidate_sha256"]:
        raise ValueError("Protected candidate identity differs")
    final = root / "runs/model_validation" / plan["model_run"] / "final"
    verify_stage(final)
    if digest(final / "candidate.joblib") != plan["candidate_sha256"]:
        raise ValueError("Candidate checksum differs from preregistration")
    encoder = root / "models/qwen3-embedding-0.6b" / spec["revision"]
    assets = json.loads((encoder / "manifest.json").read_text())
    official = json.loads((root / "configs/model.json").read_text())
    for name in MODEL_FILES:
        path = encoder / name
        if not path.is_file() or path.is_symlink() or digest(path) != assets.get(name):
            raise ValueError(f"Restore the exact local encoder asset before packaging: {name}")
        expected = official["files"][name]
        if expected["algorithm"] == "sha256":
            actual = digest(path)
        else:
            payload = path.read_bytes()
            actual = hashlib.sha1(
                b"blob " + str(len(payload)).encode() + b"\0" + payload
            ).hexdigest()
        if actual != expected["digest"]:
            raise ValueError(f"Encoder asset differs from the pinned upstream object: {name}")
    sources = {p.name: digest(p) for p in sorted((root / "src/jigsaw_rules").glob("*.py"))}
    files = {"candidate.joblib": final / "candidate.joblib"}
    for relative in (
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        "LICENSE",
        "configs/semantic.json",
        "configs/model.json",
        "configs/delivery.json",
        "scripts/offline_inference.py",
    ):
        files[relative] = root / relative
    files.update({f"src/jigsaw_rules/{name}": root / "src/jigsaw_rules" / name for name in sources})
    files.update(
        {
            f"models/qwen3-embedding-0.6b/{spec['revision']}/{name}": encoder / name
            for name in [*MODEL_FILES, "manifest.json"]
        }
    )
    manifest = {
        "schema": 1,
        "kind": "accepted_postcompetition_route",
        "candidate_sha256": plan["candidate_sha256"],
        "model_run": plan["model_run"],
        "prediction_commit": evidence["metadata"]["prediction_commit"],
        "protected_results_sha256": digest(root / "reports/confirmation/results.json"),
        "packages": software_versions(),
        "sources": sources,
        "files": {name: digest(path) for name, path in files.items()},
        "data_scope": (
            "Original 2,029 rows plus 9,106 post-competition development rows; "
            "no protected targets used for fitting"
        ),
        "missing_support": "Reject incomplete inputs before model loading",
    }
    target = destination / content_key(manifest)[:20]
    destination.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination / "package.lock"), timeout=1):
        if target.exists():
            if json.loads((target / "bundle.json").read_text()) != manifest:
                raise ValueError("Different existing offline bundle; refusing overwrite")
            verify_bundle(target, digest(target / "bundle.json"))
            return target
        with tempfile.TemporaryDirectory(prefix=".bundle-", dir=destination) as temporary:
            work = Path(temporary) / "artifacts"
            work.mkdir()
            with Progress(destination / "events.jsonl", "offline_package") as log:
                for name, source in files.items():
                    output = work / name
                    output.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, output)
                    if digest(output) != manifest["files"][name]:
                        raise ValueError("Artifact changed during packaging")
                    log.emit("file_packaged", file=name, bytes=output.stat().st_size)
                atomic_json(work / "bundle.json", manifest)
                verify_bundle(work, digest(work / "bundle.json"))
                os.replace(work, target)
    return target


class OfflineEncoder(QwenEncoder):
    """Preserve encoding math while loading verified local files without download fallbacks."""

    def _load(self, log: Progress) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(self.spec["threads"])
        torch.manual_seed(self.spec["seed"])
        torch.use_deterministic_algorithms(True)
        directory = self.root / "models/qwen3-embedding-0.6b" / self.spec["revision"]
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
        log.emit("offline_model_loaded", revision=self.spec["revision"], device="cpu")


class OfflineModel:
    def __init__(self, bundle: Path, expected_sha256: str, cache: Path):
        self.bundle, self.cache = Path(bundle), Path(cache)
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        self.manifest = verify_bundle(self.bundle, expected_sha256)
        self.expected_sha256 = expected_sha256
        self.spec = load_spec(self.bundle)
        self.candidate = None
        self.encoder = OfflineEncoder(self.bundle, self.spec)

    def _candidate(self):
        if self.candidate is None:
            # Verify immediately before deserialization, even after a lazy wait.
            if digest(self.bundle / "candidate.joblib") != self.manifest["candidate_sha256"]:
                raise ValueError("Serialized candidate changed before loading")
            self.candidate = joblib.load(self.bundle / "candidate.joblib")
        return self.candidate

    def predict_vectors(self, frame: pd.DataFrame, vectors: np.ndarray) -> np.ndarray:
        validate_frame(frame, train=False)
        return self._candidate().predict(frame, vectors)

    def predict(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        validate_frame(frame, train=False)
        with FileLock(str(self.cache.with_suffix(".lock")), timeout=1):
            with Progress(self.cache / "events.jsonl", "offline_inference") as log:
                vectors, cache = encode_cached(
                    self.cache, input_texts(frame, self.spec), self.encoder, shard_size=64
                )
                vectors = vectors.reshape(len(frame), 5, -1)
                prediction = self.predict_vectors(frame, vectors)
                similarity = np.einsum("nd,nkd->nk", vectors[:, 0], vectors[:, 1:])
                closest = similarity.argmax(axis=1)
                result = pd.DataFrame(
                    {"row_id": frame.row_id.to_numpy(), "rule_violation": prediction}
                )
                validate_submission(result, frame[["row_id"]].reset_index(drop=True))
                details = {
                    "cache": cache,
                    "familiar": familiar_mask(frame, self._candidate().familiar.policies_).tolist(),
                    "nearest_support_column": [EXAMPLES[i] for i in closest],
                    "nearest_support_similarity": similarity.max(axis=1).astype(float).tolist(),
                }
                log.emit("predicted", rows=len(result), **cache)
                return result, details


def predict_file(
    bundle: Path,
    expected_sha256: str,
    inputs: Path,
    output: Path,
    cache: Path,
    *,
    batch_rows: int = 32,
) -> Path:
    """Create a benchmark prediction file; never fit, read labels, or submit to Kaggle."""
    if type(batch_rows) is not int or batch_rows < 1:
        raise ValueError("batch_rows must be a positive integer")
    payload = inputs.read_bytes()
    frame = pd.read_csv(io.BytesIO(payload))
    validate_frame(frame, train=False)
    identity = {
        "bundle_sha256": expected_sha256,
        "inputs_sha256": hashlib.sha256(payload).hexdigest(),
        "batch_rows": batch_rows,
        "packages": software_versions(),
    }
    directory = cache / "predictions" / content_key(identity)[:20]
    model = OfflineModel(bundle, expected_sha256, cache / "encoding")
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / "inference.lock"), timeout=1):
        with Progress(directory / "events.jsonl", "offline_file") as log:
            blocks = []
            for start in range(0, len(frame), batch_rows):
                end = min(start + batch_rows, len(frame))

                def action(path: Path, first=start, last=end) -> None:
                    scores, details = model.predict(frame.iloc[first:last])
                    atomic_bytes(path / "predictions.csv", scores.to_csv(index=False).encode())
                    atomic_json(path / "details.json", details)

                name = f"batch_{start:09d}"
                if (directory / name).exists():
                    verify_stage(directory / name)
                saved = stage(directory, name, action)
                scores = pd.read_csv(saved / "predictions.csv")
                validate_submission(
                    scores, frame.iloc[start:end][["row_id"]].reset_index(drop=True)
                )
                blocks.append(scores)
                log.emit("batch_completed", completed=end, total=len(frame))
            combined = pd.concat(blocks, ignore_index=True)
            validate_submission(combined, frame[["row_id"]].reset_index(drop=True))
            if digest(inputs) != identity["inputs_sha256"]:
                raise ValueError("Inputs changed during inference; export refused")
            atomic_bytes(output / "predictions.csv", combined.to_csv(index=False).encode())
            atomic_json(
                output / "prediction_manifest.json",
                {
                    **identity,
                    "rows": len(frame),
                    "prediction_sha256": digest(output / "predictions.csv"),
                    "scope": "Accepted post-competition benchmark model; no Kaggle submission",
                },
            )
    return output / "predictions.csv"
