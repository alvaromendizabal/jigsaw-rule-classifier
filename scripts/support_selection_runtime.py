"""Small CPU audit I/O layer: atomic manifests, bounded progress, and safe replay."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest(path: Path, tick=lambda: None) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected a regular, nonsymlink artifact")
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            tick()
            result.update(chunk)
    return result.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("Refusing to replace a symlink")
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(path, json_bytes(payload))


def safe_member(root: Path, name: str) -> Path:
    if not isinstance(name, str) or "\\" in name or ":" in name:
        raise ValueError("Unsafe checkpoint member")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Unsafe checkpoint member")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Checkpoint member escapes its root")
    if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
        raise ValueError("Symlink in checkpoint path")
    return path


def save_stage(
    root: Path, contract_id: str, payloads: dict[str, bytes], metadata: dict | None = None
) -> dict:
    files = {}
    for name, payload in payloads.items():
        path = safe_member(root, name)
        atomic_bytes(path, payload)
        files[name] = {"bytes": len(payload), "sha256": digest(path)}
    marker = {"schema": 1, "contract_id": contract_id, "files": files, "metadata": metadata or {}}
    atomic_json(root / "complete.json", marker)  # Manifest is always last.
    load_stage(root, contract_id, set(payloads))  # Actual read-back verification.
    return marker


def load_stage(
    root: Path, contract_id: str, expected_files: set[str] | None = None, tick=lambda: None
) -> dict | None:
    marker = root / "complete.json"
    if not marker.exists():
        return None
    if marker.is_symlink():
        raise ValueError("Symlink completion marker")
    record = json.loads(marker.read_text(encoding="utf-8"))
    if record.get("schema") != 1 or record.get("contract_id") != contract_id:
        raise ValueError("Checkpoint contract mismatch")
    files = record.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Missing checkpoint inventory")
    if expected_files is not None and set(files) != expected_files:
        raise ValueError("Checkpoint inventory differs")
    for name, pin in files.items():
        path = safe_member(root, name)
        if path.stat().st_size != pin["bytes"] or digest(path, tick) != pin["sha256"]:
            raise ValueError("Checkpoint corruption: preserve it and investigate")
    return record


def checked_input(path: Path, pin: dict, tick=lambda: None) -> dict:
    if path.stat().st_size != pin["bytes"]:
        raise ValueError("Pinned input size differs; no automatic recomputation")
    value = digest(path, tick)
    if value != pin["sha256"]:
        raise ValueError("Pinned input checksum differs; preserve and investigate")
    return {"bytes": path.stat().st_size, "sha256": value}


class Progress:
    def __init__(self, path: Path, max_seconds: float, heartbeat_seconds: float):
        self.path = path
        self.max_seconds = max_seconds
        self.interval = heartbeat_seconds
        self.started = self.last_progress = time.monotonic()
        self.stage = "initializing"
        self.completed_rows = 0
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.emit("started", max_seconds=self.max_seconds, python=platform.python_version())
        self.thread.start()
        return self

    def __exit__(self, kind, value, tb):
        self.stop.set()
        self.thread.join(timeout=2)
        self.emit(
            "finished" if kind is None else "stopped", error_type=kind.__name__ if kind else None
        )
        return False

    def check(self):
        if time.monotonic() - self.started >= self.max_seconds:
            raise TimeoutError("Audit time cap reached; completed checkpoints are preserved")

    def advance(self, stage: str, completed_rows: int | None = None):
        self.check()
        with self.lock:
            self.stage = stage
            self.last_progress = time.monotonic()
            if completed_rows is not None:
                self.completed_rows = completed_rows
        self.emit("progress")

    def emit(self, event: str, **fields):
        with self.lock:
            record = {
                "utc": utc(),
                "event": event,
                "stage": self.stage,
                "completed_rows": self.completed_rows,
                "elapsed_seconds": round(time.monotonic() - self.started, 3),
                "seconds_since_progress": round(time.monotonic() - self.last_progress, 3),
                **fields,
            }
            line = json.dumps(record, sort_keys=True, allow_nan=False)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
            print(line, flush=True)

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            self.emit("heartbeat")
