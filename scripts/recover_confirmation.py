"""Verify a downloaded protected archive and restore predictions without changing seed caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path, PurePosixPath

from jigsaw_rules.final_model import verify_stage
from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest

ROOT = Path(__file__).resolve().parents[1]
PREDICTION_FILES = {"predictions.csv", "provenance.json", "audit.json", "complete.json"}
PREFIXES = (
    "runs/embeddings/",
    "runs/expanded_embeddings/",
    "runs/confirmation/",
    "reports/confirmation/",
    "logs/",
)


def recover(root: Path, archive: Path, run_id: str, expected_sha: str, expected_bytes: int) -> dict:
    """Validate every archived stage before any write; preserve all unrelated local files."""
    if not re.fullmatch(r"[0-9a-f]{20}", run_id):
        raise ValueError("Invalid protected run identifier")
    if archive.stat().st_size != expected_bytes or digest(archive) != expected_sha:
        raise ValueError("Downloaded protected archive differs from the completed worker state")
    prediction_prefix = f"runs/confirmation/{run_id}/predictions/"
    checksums, markers, predictions = {}, {}, {}
    with tarfile.open(archive) as stream:
        for item in stream:
            path = PurePosixPath(item.name)
            if (
                not item.isfile()
                or item.name in checksums
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in item.name
                or path.as_posix() != item.name
                or not item.name.startswith(PREFIXES)
                or path.name in {"solution.csv", "target_access.json"}
                or "evaluation" in path.parts
            ):
                raise ValueError("Unsafe, duplicate or target-bearing archive member")
            value = hashlib.sha256()
            payload = bytearray()
            keep = item.name.startswith(prediction_prefix) or path.name == "complete.json"
            handle = stream.extractfile(item)
            if handle is None:
                raise ValueError("Unreadable archive member")
            while block := handle.read(1024 * 1024):
                value.update(block)
                if keep:
                    payload.extend(block)
            checksums[item.name] = value.hexdigest()
            if path.name == "complete.json":
                markers[item.name] = json.loads(payload)
            if item.name.startswith(prediction_prefix):
                name = item.name.removeprefix(prediction_prefix)
                if name not in PREDICTION_FILES:
                    raise ValueError("Unexpected protected prediction artifact")
                predictions[name] = bytes(payload)
    if set(predictions) != PREDICTION_FILES:
        raise ValueError("Protected prediction stage is incomplete")
    for name, record in markers.items():
        members = record.get("files", {})
        if not members:
            raise ValueError("Empty protected archive stage")
        parent = PurePosixPath(name).parent
        for member, value in members.items():
            path = PurePosixPath(member)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe protected stage member")
            if checksums.get((parent / path).as_posix()) != value:
                raise ValueError("Protected archive stage checksum differs")
    record = markers[prediction_prefix + "complete.json"]
    if set(record["files"]) != PREDICTION_FILES - {"complete.json"}:
        raise ValueError("Protected prediction stage schema differs")
    destination = root / prediction_prefix
    for name, payload in predictions.items():
        path = destination / name
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
            or (path.exists() and path.read_bytes() != payload)
        ):
            raise ValueError("Recovery would replace different existing protected predictions")
    # Commit the completion marker last, after all members were validated and written.
    for name in [*sorted(PREDICTION_FILES - {"complete.json"}), "complete.json"]:
        path = destination / name
        if not path.exists():
            atomic_bytes(path, predictions[name])
    verify_stage(destination)
    return {
        "schema": 1,
        "run_id": run_id,
        "archive_sha256": expected_sha,
        "archive_bytes": expected_bytes,
        "verified_archive_members": len(checksums),
        "verified_completed_stages": len(markers),
        "prediction_stage_sha256": digest(destination / "complete.json"),
        "restored_files": sorted(prediction_prefix + name for name in predictions),
        "embedding_caches_modified": False,
        "reserved_targets_accessed": False,
        "scope": "Archive and prediction-stage recovery; run freeze to verify model/row lineage",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--bytes", type=int, required=True)
    args = parser.parse_args()
    with Progress(ROOT / "logs/confirmation-recovery.jsonl", "confirmation_recovery") as log:
        record = recover(ROOT, args.archive, args.run_id, args.sha256, args.bytes)
        atomic_json(ROOT / "reports/checkpoints/protected_recovery.json", record)
        log.emit("verified", **record)
