"""Atomic stage commits, checksummed reuse, and UTC progress events."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from filelock import FileLock


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def environment() -> dict:
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "plotly"]
    return {"python": platform.python_version(), "packages": {p: version(p) for p in packages}}


def fingerprint(data_dir: Path, config: dict) -> tuple[str, dict]:
    source = Path(__file__).parent
    record = {
        "data": {p.name: digest(p) for p in sorted(data_dir.glob("*.csv"))},
        "code": {p.name: digest(p) for p in sorted(source.glob("*.py"))},
        "config": config,
        "environment": environment(),
    }
    key = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:20]
    return key, record


class Progress:
    """Every long operation emits start, heartbeat, finish/failure, and elapsed seconds."""

    def __init__(self, path: Path, stage: str, heartbeat_seconds: float = 15):
        self.path = path
        self.stage = stage
        self.interval = heartbeat_seconds
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.guard = threading.Lock()
        self.thread: threading.Thread | None = None

    def emit(self, event: str, **fields: Any) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "stage": self.stage,
            "event": event,
            "elapsed_seconds": round(time.monotonic() - self.started, 3),
            **fields,
        }
        line = json.dumps(record, allow_nan=False)
        with self.guard:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
            print(line, flush=True)

    def _heartbeat(self) -> None:
        while not self.stop.wait(self.interval):
            self.emit("heartbeat")

    def __enter__(self) -> Progress:
        self.emit("started")
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, kind, error, traceback) -> None:
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=2)
        self.emit("failed" if error else "completed", error_type=kind.__name__ if kind else None)


def stage(directory: Path, name: str, action: Callable[[Path], None]) -> Path:
    """A failed stage restarts; completed, intact stages are reused without recomputing."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    with FileLock(str(directory / f"{name}.lock"), timeout=1):
        with Progress(directory / "events.jsonl", name) as log:
            marker = target / "complete.json"
            if marker.exists():
                record = json.loads(marker.read_text())
                intact = bool(record["files"]) and all(
                    (target / p).is_file() and digest(target / p) == sha
                    for p, sha in record["files"].items()
                )
                if intact:
                    log.emit("reused", files=len(record["files"]))
                    return target
                log.emit("recomputing", reason="output checksum mismatch")
            # Only committed directories count as checkpoints. Abandoned work is isolated.
            with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=directory) as temporary:
                work = Path(temporary) / "artifacts"
                work.mkdir()
                action(work)
                files = {
                    str(p.relative_to(work)): digest(p)
                    for p in sorted(work.rglob("*"))
                    if p.is_file() and not p.name.endswith(".partial")
                }
                if not files:
                    raise ValueError(f"Stage {name} produced no artifacts")
                atomic_json(
                    work / "complete.json",
                    {"files": files, "finished_at": datetime.now(UTC).isoformat()},
                )
                if target.exists():
                    shutil.rmtree(target)
                os.replace(work, target)
            return target
