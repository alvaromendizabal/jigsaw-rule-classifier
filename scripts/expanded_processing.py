"""Bounded CPU worker for four-policy research; checkpoints committed stages to S3."""

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

ACCOUNT = "560403859723"
BUCKET = "sagemaker-jigsaw-rules-560403859723-us-west-2"
SNAPSHOT = "8762046fe72017d8567c9c56e68f82db95ba93e1edf2c4d33ee3c4744fee724f"
RELEASE_KEY = "experiments/released-boundary-20260908T2230/checkpoints/released-boundary.tar.gz"
RELEASE_SHA = "0b809ee83a0925d77be5f11e29dc60e5af46e6523366b24348bb5259d41ab967"
ROOTS = ("runs/embeddings/", "runs/expanded_embeddings/", "runs/expanded/")


def restorable_objects(entries: list[dict], prefix: str) -> list[dict]:
    """Ignore an interrupted upload until its stage completion marker exists."""
    keys = {item["Key"] for item in entries}
    if any(not key.startswith(prefix) for key in keys):
        raise ValueError("Checkpoint entry is outside the requested prefix")
    result = []
    for item in entries:
        relative = item["Key"][len(prefix) :]
        path = Path(relative)
        marker = prefix + path.parent.as_posix() + "/complete.json"
        standalone = len(path.parts) == 4 and path.name in {
            "contract.json",
            "provenance.json",
            "completed.json",
        }
        if marker in keys or standalone:
            result.append(item)
    return result


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def extract(archive: Path, root: Path, prefixes: tuple[str, ...] = ()) -> None:
    with tarfile.open(archive) as stream:
        members = []
        for item in stream.getmembers():
            if prefixes and not item.name.startswith(prefixes):
                continue
            destination = (root / item.name).resolve()
            if not item.isfile() or not destination.is_relative_to(root):
                raise ValueError("Archive must contain only safe regular files")
            members.append(item)
        stream.extractall(root, members=members)


def main() -> None:
    bucket = os.environ["JIGSAW_BUCKET"]
    prefix = os.environ["JIGSAW_PREFIX"].rstrip("/")
    if bucket != BUCKET or not prefix.startswith("experiments/expanded-features-"):
        raise ValueError("Worker is limited to the owned project experiment prefix")
    root = Path("/opt/ml/processing/work/jigsaw")
    root.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", region_name="us-west-2")
    expected = {"ExpectedBucketOwner": ACCOUNT}
    encrypted = {**expected, "ServerSideEncryption": "AES256"}
    started = time.monotonic()

    def emit(event: str, **fields) -> None:
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": event,
            "total_elapsed_seconds": time.monotonic() - started,
            **fields,
        }
        print(json.dumps(record), flush=True)
        s3.put_object(
            Bucket=bucket, Key=prefix + "/state.json", Body=json.dumps(record).encode(), **encrypted
        )

    def upload(path: Path, key: str) -> None:
        s3.upload_file(str(path), bucket, key, ExtraArgs=encrypted)

    def download(key: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(path), ExtraArgs=expected)

    subprocess.run([sys.executable, "-m", "pip", "install", "uv==0.11.33"], check=True)
    source = Path("/opt/ml/processing/input/code/source.tar.gz")
    if sha256(source) != os.environ["JIGSAW_SOURCE_SHA256"]:
        raise ValueError("Worker source checksum mismatch")
    extract(source, root)
    subprocess.run(
        ["uv", "sync", "--locked", "--python", "3.12.13", "--extra", "semantic", "--group", "dev"],
        cwd=root,
        check=True,
    )
    python = str(root / ".venv/bin/python")
    emit("environment_ready")
    snapshot = root.parent / "snapshot.json"
    download("snapshots/" + SNAPSHOT + ".json", snapshot)
    if sha256(snapshot) != SNAPSHOT:
        raise ValueError("Original snapshot checksum differs")
    files = {
        name: sha
        for name, sha in json.loads(snapshot.read_text())["files"].items()
        if name.startswith(("data/raw/", "runs/embeddings/"))
    }

    def restore(item) -> None:
        name, sha = item
        path = root / name
        if not path.resolve().is_relative_to(root):
            raise ValueError("Unsafe snapshot path")
        download("objects/" + sha, path)
        if sha256(path) != sha:
            raise ValueError("Restored object checksum mismatch")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(restore, files.items()))
    release = root.parent / "released-boundary.tar.gz"
    download(RELEASE_KEY, release)
    if sha256(release) != RELEASE_SHA:
        raise ValueError("Released boundary checkpoint checksum mismatch")
    # The source archive contains solution.csv only inside release.zip. Do not
    # extract that source archive; restore only the verified research checkpoint.
    extract(release, root, ("runs/released/",))
    emit("research_inputs_restored", snapshot_paths=len(files))
    resume = os.environ.get("JIGSAW_RESUME_PREFIX")
    if resume:
        if not resume.startswith("experiments/expanded-features-"):
            raise ValueError("Resume must use a project expanded-study checkpoint")
        checkpoint_prefix = resume.rstrip("/") + "/checkpoint/"
        entries = []
        for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=checkpoint_prefix, **expected
        ):
            entries.extend(page.get("Contents", []))
        listed_count = len(entries)
        entries = restorable_objects(entries, checkpoint_prefix)

        def restore_checkpoint(item) -> None:
            relative = item["Key"][len(checkpoint_prefix) :]
            destination = root / relative
            if not relative.startswith(ROOTS) or not destination.resolve().is_relative_to(root):
                raise ValueError("Unsafe resumed checkpoint path")
            if destination.exists():
                temporary = destination.with_name(destination.name + ".resume-check")
                download(item["Key"], temporary)
                if sha256(temporary) != sha256(destination):
                    raise ValueError("Resume would overwrite a different preserved artifact")
                temporary.unlink()
                return
            download(item["Key"], destination)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(restore_checkpoint, entries))
        emit(
            "prior_checkpoint_restored",
            files=len(entries),
            uncommitted_objects_ignored=listed_count - len(entries),
        )
    originals = {p.relative_to(root).as_posix() for p in (root / "runs").rglob("complete.json")}
    # Resumed stages need copying to this run's prefix; the original 32 embedding
    # shards remain in the pinned source snapshot and are never duplicated.
    originals = {name for name in originals if name.startswith("runs/released/") or name in files}
    uploaded, errors = set(), []
    stop = threading.Event()

    def checkpoint() -> None:
        markers = [p for prefix_path in ROOTS for p in (root / prefix_path).rglob("complete.json")]
        for marker in markers:
            relative = marker.relative_to(root).as_posix()
            if relative in originals:
                continue
            marker_sha = sha256(marker)
            if (relative, marker_sha) in uploaded:
                continue
            for name, sha in json.loads(marker.read_text())["files"].items():
                path = marker.parent / name
                if not path.resolve().is_relative_to(marker.parent) or sha256(path) != sha:
                    raise ValueError("Active checkpoint checksum mismatch")
                upload(path, prefix + "/checkpoint/" + path.relative_to(root).as_posix())
            # A stage becomes resumable only after every verified member is uploaded.
            upload(marker, prefix + "/checkpoint/" + relative)
            uploaded.add((relative, marker_sha))
        for relative_root in ROOTS:
            for path in (root / relative_root).glob("*/*.json"):
                if path.name in {"contract.json", "provenance.json", "completed.json"}:
                    upload(path, prefix + "/checkpoint/" + path.relative_to(root).as_posix())
        for path in (root / "logs").glob("expanded*.jsonl"):
            upload(path, prefix + "/logs/" + path.name)
        emit("checkpoint", completed_stages=len(uploaded))

    def periodic() -> None:
        while not stop.wait(45):
            try:
                checkpoint()
            except Exception as error:
                errors.append(type(error).__name__)
                emit("checkpoint_failed", error_type=type(error).__name__)
                return

    thread = threading.Thread(target=periodic, daemon=True)
    thread.start()
    status = 1
    try:
        emit("expanded_feature_study_started")
        process = subprocess.run([python, "scripts/run_expanded.py", "--encode"], cwd=root)
        status = process.returncode
    finally:
        stop.set()
        thread.join()
        checkpoint()
        for path in (root / "reports/expanded").glob("*.json"):
            upload(path, prefix + "/public/reports/expanded/" + path.name)
        if status == 0 and not errors:
            archive = root.parent / "expanded-study.tar.gz"
            with tarfile.open(archive, "w:gz") as stream:
                for directory in [
                    *(root / p for p in ROOTS),
                    root / "reports/expanded",
                    root / "logs",
                ]:
                    for path in sorted(directory.rglob("*")):
                        if path.is_file() and not path.name.endswith((".partial", ".lock")):
                            stream.add(path, arcname=path.relative_to(root), recursive=False)
            upload(archive, prefix + "/checkpoints/expanded-study.tar.gz")
            emit(
                "completed",
                archive_sha256=sha256(archive),
                archive_bytes=archive.stat().st_size,
                completed_stages=len(uploaded),
            )
        else:
            emit("failed", exit_code=status, checkpoint_errors=errors)
    if status or errors:
        raise SystemExit(status or 1)


if __name__ == "__main__":
    main()
