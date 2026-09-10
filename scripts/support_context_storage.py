"""Object-scoped HF transport with atomic, hash-checked checkpoint snapshots."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, path, expected=None, *, missing_ok=False):
    """Never include a bearer URL in an exception or log."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=30) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out, length=4 * 1024 * 1024)
        if expected is not None and sha256(temporary) != expected:
            raise ValueError("Pinned input checksum differs")
        temporary.replace(path)
        return True
    except urllib.error.HTTPError as error:
        if missing_ok and error.code == 404:
            return False
        raise RuntimeError(f"Scoped download HTTP {error.code}") from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError("Scoped download connection failed") from None
    finally:
        temporary.unlink(missing_ok=True)


class SignedCheckpointStore:
    """Minimal S3-shaped interface; no AWS credentials, listing or arbitrary keys.

    A shard's features and completion marker are committed together in one S3
    PUT. Completed snapshots survive interruption; S3 versions retain history.
    """

    def __init__(self, bucket, inputs, put_url, get_url, deadline, directory):
        self.bucket = bucket
        self.inputs = {x["key"]: x for x in inputs}
        self.put_url, self.get_url = put_url, get_url
        self.deadline = float(deadline)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prefix = None
        self.files = {}
        self.last_snapshot_sha256 = None

    def guard(self):
        if time.time() > self.deadline - 60:
            raise TimeoutError("URL expiry approaching; stop with completed shards saved")

    def get_paginator(self, name):
        if name != "list_objects_v2":
            raise ValueError("Unsupported storage operation")
        return self

    def paginate(self, *, Bucket, Prefix):
        if Bucket != self.bucket or not Prefix.startswith(
            "experiments/support-context-20260910/checkpoints/"
        ):
            raise ValueError("Checkpoint scope differs")
        self.prefix = Prefix
        self.guard()
        path = self.directory / "restore.tar.gz"
        if download(self.get_url, path, missing_ok=True):
            with tarfile.open(path) as archive:
                manifest = json.load(archive.extractfile("snapshot.json"))
                if manifest["schema"] != 1 or manifest["prefix"] != Prefix:
                    raise ValueError("Existing snapshot uses another runtime contract")
                expected = manifest["files"]
                if len(archive.getmembers()) != len(expected) + 1:
                    raise ValueError("Unexpected snapshot members")
                for name, checksum in expected.items():
                    target = self.directory / "restored" / name
                    if not target.resolve().is_relative_to((self.directory / "restored").resolve()):
                        raise ValueError("Unsafe checkpoint path")
                    member = archive.getmember(name)
                    if not member.isfile():
                        raise ValueError("Snapshot links are forbidden")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source, target.open("wb") as out:
                        shutil.copyfileobj(source, out)
                    if sha256(target) != checksum:
                        raise ValueError("Saved checkpoint checksum differs")
                    self.files[Prefix + name] = target
            print(json.dumps({"stage": "snapshot_restored", "files": len(self.files)}), flush=True)
        yield {"Contents": [{"Key": k} for k in sorted(self.files)]}

    def download_file(self, bucket, key, filename):
        if bucket != self.bucket:
            raise ValueError("Input bucket differs")
        self.guard()
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        if key in self.inputs:
            item = self.inputs[key]
            download(item["url"], path, item["sha256"])
        elif key in self.files:
            shutil.copyfile(self.files[key], path)
        else:
            raise ValueError("Input object is outside the authorized scope")

    def upload_file(self, filename, bucket, key, ExtraArgs):
        if (
            bucket != self.bucket
            or self.prefix is None
            or not key.startswith(self.prefix)
            or ExtraArgs != {"ServerSideEncryption": "AES256"}
        ):
            raise ValueError("Output object is outside the authorized scope")
        self.guard()
        self.files[key] = Path(filename)
        # Commit each completed shard/fold atomically; events join the next save.
        if key.endswith("/complete.json") or key == self.prefix + "contract.json":
            self.flush()

    def flush(self):
        if not self.files:
            return
        if time.time() >= self.deadline:
            raise TimeoutError("Signed checkpoint URL expired")
        relative = {k[len(self.prefix) :]: v for k, v in self.files.items()}
        manifest = {
            "schema": 1,
            "prefix": self.prefix,
            "files": {n: sha256(p) for n, p in sorted(relative.items())},
        }
        path = self.directory / "checkpoints.tar.gz"
        with tarfile.open(path, "w:gz", compresslevel=1) as archive:
            for name, source in sorted(relative.items()):
                archive.add(source, arcname=name, recursive=False)
            payload = json.dumps(manifest, sort_keys=True).encode()
            info = tarfile.TarInfo("snapshot.json")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        payload = path.read_bytes()
        request = urllib.request.Request(
            self.put_url,
            data=payload,
            method="PUT",
            headers={"x-amz-server-side-encryption": "AES256"},
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                if response.status != 200:
                    raise RuntimeError("Checkpoint PUT failed")
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"Checkpoint PUT HTTP {error.code}") from None
        except (urllib.error.URLError, TimeoutError):
            raise RuntimeError("Checkpoint PUT connection failed") from None
        self.last_snapshot_sha256 = hashlib.sha256(payload).hexdigest()
        print(
            json.dumps(
                {
                    "stage": "checkpoint_saved",
                    "files": len(relative),
                    "bytes": len(payload),
                    "sha256": self.last_snapshot_sha256,
                }
            ),
            flush=True,
        )
