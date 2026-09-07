"""Content-addressed S3 snapshots with complete-workspace and conditional-write guards."""

from __future__ import annotations

import hashlib
import io
import json
import re
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from filelock import FileLock

from jigsaw_rules.runtime import Progress, atomic_bytes, digest


def client(region: str):
    return boto3.client(
        "s3",
        region_name=region,
        config=Config(
            signature_version="s3v4",
            retries={"total_max_attempts": 5, "mode": "standard"},
            connect_timeout=10,
            read_timeout=60,
        ),
    )


def manifest_files(manifest: dict) -> dict[str, str]:
    """Validate the complete namespace before performing a read or write."""
    if not isinstance(manifest, dict) or manifest.get("schema") != 1:
        raise ValueError("Unsupported backup schema")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Backup manifest must contain files")
    for name, sha in files.items():
        if not isinstance(name, str):
            raise ValueError("Unsafe backup path")
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or relative.as_posix() != name
            or not name.startswith(("data/raw/", "runs/"))
        ):
            raise ValueError("Unsafe backup path")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError("Invalid content hash")
    return files


def _local_files(root: Path) -> dict[str, Path]:
    files = {}
    for folder in ("data/raw", "runs"):
        directory = root / folder
        if not directory.resolve().is_relative_to(root):
            raise ValueError("Backup source escapes project through a symlink")
        for path in directory.rglob("*"):
            relative = path.relative_to(root)
            if (
                not path.is_file()
                or path.is_symlink()
                or any(part.startswith(".") for part in relative.parts)
                or any(part.endswith((".lock", ".partial")) for part in relative.parts)
            ):
                continue
            if not path.resolve().is_relative_to(root):
                raise ValueError("Backup source escapes project through a symlink")
            files[relative.as_posix()] = path
    return files


def _latest(s3, bucket: str) -> tuple[dict[str, str], str | None]:
    try:
        response = s3.get_object(Bucket=bucket, Key="latest.json")
    except ClientError as error:
        # A missing key is a new backup. Missing buckets, denied access, and
        # network failures must never be mistaken for an empty remote history.
        if error.response["Error"]["Code"] == "NoSuchKey":
            return {}, None
        raise
    with response["Body"] as stream:
        files = manifest_files(json.load(stream))
    etag = response.get("ETag")
    if not isinstance(etag, str) or not etag:
        raise ValueError("Latest snapshot has no ETag; conditional update is unavailable")
    return files, etag


def backup(root: Path, bucket: str, region: str, s3=None) -> str:
    """Never publish a snapshot that omits prior paths or overwrites another writer.

    Call between model stages, not alongside another training or restore command.
    JSONL logs are copied as byte snapshots and may continue appending. Immutable
    artifacts must remain unchanged until the conditional manifest commit.
    """
    root = root.resolve()
    s3 = s3 or client(region)
    files = _local_files(root)
    if not files:
        raise ValueError("No data/raw or runs files to back up")
    (root / "runs").mkdir(exist_ok=True)
    with (
        FileLock(str(root / "runs/cloud_backup.lock"), timeout=1),
        Progress(root / "logs/cloud.jsonl", "cloud_backup") as log,
    ):
        previous, etag = _latest(s3, bucket)
        files = _local_files(root)
        missing = sorted(previous.keys() - files.keys())
        if missing:
            raise FileNotFoundError(
                f"Backup would omit {len(missing)} existing snapshot paths. "
                "Restore the saved workspace before backing up; latest.json is unchanged. "
                f"First missing path: {missing[0]}"
            )
        log.emit("snapshot_preflight_passed", previous_files=len(previous), local_files=len(files))
        manifest = {"schema": 1, "created_at": datetime.now(UTC).isoformat(), "files": {}}
        for name, path in sorted(files.items()):
            payload = path.read_bytes()
            sha = hashlib.sha256(payload).hexdigest()
            key = f"objects/{sha}"
            found = s3.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1).get("Contents", [])
            if not any(obj["Key"] == key for obj in found):
                s3.upload_fileobj(
                    io.BytesIO(payload),
                    bucket,
                    key,
                    ExtraArgs={"ServerSideEncryption": "AES256", "Metadata": {"sha256": sha}},
                )
            manifest["files"][name] = sha
            log.emit("file_snapshotted", path=name)
        # Checkpoint contents are immutable. Do not commit a mixed snapshot if
        # another process replaced a model/data file while its upload was in flight.
        if _local_files(root).keys() != files.keys() or any(
            not path.is_file()
            or (path.suffix != ".jsonl" and digest(path) != manifest["files"][name])
            for name, path in files.items()
        ):
            raise RuntimeError("Workspace changed during backup; latest.json is unchanged")
        content = json.dumps(manifest, sort_keys=True).encode()
        snapshot = hashlib.sha256(content).hexdigest()
        s3.upload_fileobj(
            io.BytesIO(content),
            bucket,
            f"snapshots/{snapshot}.json",
            ExtraArgs={"ServerSideEncryption": "AES256"},
        )
        condition = {"IfMatch": etag} if etag is not None else {"IfNoneMatch": "*"}
        try:
            s3.put_object(
                Bucket=bucket,
                Key="latest.json",
                Body=content,
                ServerSideEncryption="AES256",
                ContentType="application/json",
                **condition,
            )
        except ClientError as error:
            if error.response["Error"]["Code"] in {
                "PreconditionFailed",
                "ConditionalRequestConflict",
                "412",
                "409",
            }:
                raise RuntimeError(
                    "Another backup changed latest.json; its snapshot was preserved. "
                    "Reconcile or restore that workspace before retrying."
                ) from error
            raise
        log.emit("snapshot_committed", snapshot=snapshot, files=len(files))
        return snapshot


def restore(root: Path, bucket: str, region: str, s3=None) -> int:
    root = root.resolve()
    s3 = s3 or client(region)
    with s3.get_object(Bucket=bucket, Key="latest.json")["Body"] as stream:
        files = manifest_files(json.load(stream))
    targets = []
    for name, sha in files.items():
        path = root / PurePosixPath(name)
        if not path.resolve().is_relative_to(root):
            raise ValueError("Backup target escapes project through a symlink")
        if path.exists() and digest(path) != sha:
            raise FileExistsError(f"Restore would replace different local content: {name}")
        targets.append((path, sha))
    with Progress(root / "logs/cloud.jsonl", "cloud_restore") as log:
        for path, sha in targets:
            if path.exists():
                continue
            with s3.get_object(Bucket=bucket, Key=f"objects/{sha}")["Body"] as stream:
                payload = stream.read()
            if hashlib.sha256(payload).hexdigest() != sha:
                raise ValueError("Downloaded object checksum mismatch")
            atomic_bytes(path, payload)
            log.emit("restored", path=path.relative_to(root).as_posix())
    return len(targets)
