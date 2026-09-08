"""Bounded SageMaker worker for frozen NLI or instruction feature experiments.

Inputs are a reviewed source archive, its SHA-256 and existing content-addressed
project checkpoints. No Studio checkout is edited. Completed inference shards are
copied to an isolated experiment prefix with their commit marker uploaded last.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(archive, root):
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        for item in members:
            if (
                item.issym()
                or item.islnk()
                or (root / item.name).resolve() != root
                and root not in (root / item.name).resolve().parents
            ):
                raise ValueError("Unsafe archive member")
        stream.extractall(root, members=members, filter="data")


def main():
    bucket, prefix = os.environ["JIGSAW_BUCKET"], os.environ["JIGSAW_PREFIX"]
    command = os.environ.get("JIGSAW_COMMAND", "pairs")
    if command not in {"pairs", "instructions", "notebooks"}:
        raise ValueError("Unknown frozen-feature experiment")
    root = Path("/opt/ml/processing/work/jigsaw")
    root.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", region_name="us-west-2")
    started = time.monotonic()

    def emit(event, **fields):
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": event,
            "total_elapsed_seconds": time.monotonic() - started,
            **fields,
        }
        print(json.dumps(record), flush=True)
        s3.put_object(
            Bucket=bucket,
            Key=prefix + "/state.json",
            Body=json.dumps(record).encode(),
            ServerSideEncryption="AES256",
        )

    # Install the pinned modern interpreter before using archive extraction features.
    subprocess.run([sys.executable, "-m", "pip", "install", "uv==0.11.33"], check=True)
    source = Path("/opt/ml/processing/input/code/source.tar.gz")
    if sha256(source) != os.environ["JIGSAW_SOURCE_SHA256"]:
        raise ValueError("Processing source archive checksum mismatch")
    # Bootstrap archives are authored here; enforce safe regular-file paths explicitly.
    with tarfile.open(source) as stream:
        for item in stream.getmembers():
            if (
                item.issym()
                or item.islnk()
                or (root / item.name).resolve() != root
                and root not in (root / item.name).resolve().parents
            ):
                raise ValueError("Unsafe source archive")
        stream.extractall(root)
    subprocess.run(
        ["uv", "sync", "--locked", "--python", "3.12.13", "--extra", "semantic", "--group", "dev"],
        cwd=root,
        check=True,
    )
    python = str(root / ".venv/bin/python")
    emit("environment_ready")
    if command == "notebooks":
        for arguments in (["--publish"], ["--publish"], ["--synthetic"]):
            subprocess.run(
                [python, "scripts/execute_notebooks.py", *arguments], cwd=root, check=True
            )
        outputs = sorted((root / "notebooks").glob("*.ipynb"))
        outputs.append(root / "logs/notebook_execution.jsonl")
        archive = root.parent / "executed-notebooks.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            for path in outputs:
                stream.add(path, arcname=path.relative_to(root), recursive=False)
        s3.upload_file(str(archive), bucket, prefix + "/public/executed-notebooks.tar.gz")
        emit(
            "completed",
            archive_sha256=sha256(archive),
            archive_bytes=archive.stat().st_size,
            notebooks=len(outputs) - 1,
            jupyter=True,
            checkpoint_reuse=True,
        )
        return
    manifest_path = root / "snapshot.json"
    s3.download_file(
        bucket,
        "snapshots/8762046fe72017d8567c9c56e68f82db95ba93e1edf2c4d33ee3c4744fee724f.json",
        str(manifest_path),
    )
    if sha256(manifest_path) != "8762046fe72017d8567c9c56e68f82db95ba93e1edf2c4d33ee3c4744fee724f":
        raise ValueError("Original snapshot checksum differs")
    files = json.loads(manifest_path.read_text())["files"]

    def restore(item):
        name, sha = item
        path = root / name
        if not path.resolve().is_relative_to(root) or not name.startswith(("runs/", "data/raw/")):
            raise ValueError("Unsafe snapshot path")
        path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, "objects/" + sha, str(path))
        if sha256(path) != sha:
            raise ValueError("Restored object checksum mismatch")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(restore, files.items()))
    emit("original_snapshot_verified", paths=len(files))
    broad = root.parent / "broad.tar.gz"
    s3.download_file(
        bucket,
        "experiments/feature-research-20260908T1720/checkpoints/broad-research.tar.gz",
        str(broad),
    )
    if sha256(broad) != "7970504257e77152109a50ba667fa1467a1c7329176d66849c433561ab6af84d":
        raise ValueError("Broad checkpoint checksum mismatch")
    # Use the pinned 3.12 interpreter and safe extraction helper for the checkpoint.
    extraction = "import sys; from pathlib import Path; from scripts.processing import extract; "
    extraction += "extract(Path(sys.argv[1]), Path(sys.argv[2]))"
    subprocess.run([python, "-c", extraction, str(broad), str(root)], cwd=root, check=True)
    # The broad archive contains its older sources. Restore the reviewed new sources.
    subprocess.run([python, "-c", extraction, str(source), str(root)], cwd=root, check=True)
    original_directories = {p.name for p in (root / "runs").iterdir() if p.is_dir()}
    resume_prefix = os.environ.get("JIGSAW_RESUME_PREFIX")
    if resume_prefix:
        checkpoint_prefix = resume_prefix.rstrip("/") + "/checkpoint/"
        restored = 0
        for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=checkpoint_prefix
        ):
            for item in page.get("Contents", []):
                relative = item["Key"][len(checkpoint_prefix) :]
                destination = root / relative
                if not relative.startswith("runs/") or not destination.resolve().is_relative_to(
                    root
                ):
                    raise ValueError("Unsafe resumed checkpoint path")
                if destination.exists():
                    raise ValueError("Resumed checkpoint would overwrite preserved work")
                destination.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, item["Key"], str(destination))
                restored += 1
        emit("prior_checkpoint_restored", files=restored)
    subprocess.run(
        [
            python,
            "-m",
            "pytest",
            "tests/test_pairs.py",
            "tests/test_context.py",
            "tests/test_research.py",
            "tests/test_instructions.py",
            "-q",
        ],
        cwd=root,
        check=True,
    )
    subprocess.run(
        [
            python,
            "-c",
            "from pathlib import Path; import json; "
            "from jigsaw_rules.pairs import prepare_pair_model; "
            "from jigsaw_rules.instructions import prepare_instruction_model; "
            + ("prepare_pair_model" if command == "pairs" else "prepare_instruction_model")
            + f"(Path.cwd(), json.loads(Path('configs/{command}.json').read_text()))",
        ],
        cwd=root,
        check=True,
    )
    uploaded = set()
    stop = threading.Event()
    errors = []

    def checkpoint():
        paths = list((root / "runs" / command).glob("*/batch_*/complete.json"))
        paths += [
            p
            for p in (root / "runs").glob("*/*/complete.json")
            if p.parent.parent.name not in original_directories
        ]
        for marker in paths:
            key = marker.relative_to(root).as_posix()
            marker_sha = sha256(marker)
            if (key, marker_sha) in uploaded:
                continue
            for name, sha in json.loads(marker.read_text())["files"].items():
                path = marker.parent / name
                if sha256(path) != sha:
                    raise ValueError("Active checkpoint hash mismatch")
                s3.upload_file(
                    str(path), bucket, prefix + "/checkpoint/" + path.relative_to(root).as_posix()
                )
            s3.upload_file(str(marker), bucket, prefix + "/checkpoint/" + key)
            uploaded.add((key, marker_sha))
        for path in (root / "runs" / command).glob("*/contract.json"):
            s3.upload_file(
                str(path), bucket, prefix + "/checkpoint/" + path.relative_to(root).as_posix()
            )
        for path in (root / "logs").glob(f"*{command}*.jsonl"):
            s3.upload_file(str(path), bucket, prefix + "/logs/" + path.name)
        emit("inference_checkpoint", completed_stages=len(uploaded))

    def periodic():
        while not stop.wait(45):
            try:
                checkpoint()
            except Exception as error:
                errors.append(type(error).__name__)
                emit("checkpoint_error", error_type=type(error).__name__)
                return

    thread = threading.Thread(target=periodic, daemon=True)
    thread.start()
    status = 1
    try:
        emit("frozen_feature_research_started", command=command)
        result = subprocess.run([python, "-m", "jigsaw_rules.cli", command], cwd=root)
        status = result.returncode
    finally:
        stop.set()
        thread.join(timeout=60)
        checkpoint()
        for path in (root / "reports" / command).glob("*.json"):
            s3.upload_file(str(path), bucket, prefix + f"/public/reports/{command}/" + path.name)
        for directory in (root / "runs").iterdir():
            if directory.is_dir() and directory.name not in original_directories:
                for path in directory.glob("*.json"):
                    s3.upload_file(
                        str(path),
                        bucket,
                        prefix + "/checkpoint/" + path.relative_to(root).as_posix(),
                    )
        emit(
            "completed" if status == 0 and not errors else "failed",
            returncode=status,
            checkpoint_errors=errors,
        )
    if status or errors:
        raise RuntimeError("Pair experiment or durable checkpointing failed")


if __name__ == "__main__":
    main()
