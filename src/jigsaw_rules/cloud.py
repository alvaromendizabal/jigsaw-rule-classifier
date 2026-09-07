"""Content-addressed S3 snapshots. The manifest is committed after all files upload."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import boto3
from botocore.config import Config

from jigsaw_rules.runtime import Progress, atomic_bytes, digest


def client(region: str):
    return boto3.client(
        "s3",
        region_name=region,
        config=Config(
            retries={"total_max_attempts": 5, "mode": "standard"},
            connect_timeout=10,
            read_timeout=60,
        ),
    )


def backup(root: Path, bucket: str, region: str, s3=None) -> str:
    s3 = s3 or client(region)
    files = [
        p
        for folder in ["data/raw", "runs"]
        for p in (root / folder).rglob("*")
        if p.is_file()
        and not p.is_symlink()
        and not p.name.endswith((".lock", ".partial"))
        and not any(part.startswith(".") for part in p.relative_to(root).parts)
    ]
    if not files:
        raise ValueError("No data/raw or runs files to back up")
    # Snapshot is called only between stages; logs are copied as bytes before hashing.
    with Progress(root / "logs/cloud.jsonl", "cloud_backup") as log:
        manifest = {"schema": 1, "created_at": datetime.now(UTC).isoformat(), "files": {}}
        for path in sorted(files):
            payload = path.read_bytes()
            sha = hashlib.sha256(payload).hexdigest()
            key = f"objects/{sha}"
            # List by exact hash avoids treating permission errors as missing objects.
            found = s3.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1).get("Contents", [])
            if not any(obj["Key"] == key for obj in found):
                import io

                s3.upload_fileobj(
                    io.BytesIO(payload),
                    bucket,
                    key,
                    ExtraArgs={"ServerSideEncryption": "AES256", "Metadata": {"sha256": sha}},
                )
            manifest["files"][path.relative_to(root).as_posix()] = sha
            log.emit("file_verified", path=path.relative_to(root).as_posix())
        content = json.dumps(manifest, sort_keys=True).encode()
        snapshot = hashlib.sha256(content).hexdigest()
        import io

        s3.upload_fileobj(
            io.BytesIO(content),
            bucket,
            f"snapshots/{snapshot}.json",
            ExtraArgs={"ServerSideEncryption": "AES256"},
        )
        s3.upload_fileobj(
            io.BytesIO(content), bucket, "latest.json", ExtraArgs={"ServerSideEncryption": "AES256"}
        )
        log.emit("snapshot_committed", snapshot=snapshot, files=len(files))
        return snapshot


def restore(root: Path, bucket: str, region: str, s3=None) -> int:
    s3 = s3 or client(region)
    with s3.get_object(Bucket=bucket, Key="latest.json")["Body"] as stream:
        manifest = json.load(stream)
    if manifest.get("schema") != 1:
        raise ValueError("Unsupported backup schema")
    targets = []
    for name, sha in manifest["files"].items():
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or not name.startswith(("data/raw/", "runs/"))
        ):
            raise ValueError("Unsafe backup path")
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("Invalid content hash")
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()):
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
