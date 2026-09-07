"""Standard-library bootstrap so environment installation also has a heartbeat."""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path


def run(command: list[str], label: str, root: Path) -> None:
    started = time.monotonic()
    stopped = threading.Event()
    lock = threading.Lock()

    def emit(event):
        record = {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "stage": label,
            "event": event,
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }
        line = json.dumps(record)
        with lock:
            with (root / "logs/bootstrap.jsonl").open("a") as stream:
                stream.write(line + "\n")
            print(line, flush=True)

    def heartbeat():
        while not stopped.wait(15):
            emit("heartbeat")

    emit("started")
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        subprocess.run(command, cwd=root, check=True, timeout=1200)
    except BaseException:
        emit("failed")
        raise
    else:
        emit("completed")
    finally:
        stopped.set()
        thread.join(timeout=2)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    (root / "logs").mkdir(exist_ok=True)
    os.environ["PATH"] = str(Path.home() / ".local/bin") + os.pathsep + os.environ["PATH"]
    uv = shutil.which("uv")
    if not uv:
        run([sys.executable, "-m", "pip", "install", "--user", "uv==0.11.33"], "install_uv", root)
        uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv installation did not make its executable available")
    run([uv, "sync", "--locked", "--group", "dev"], "environment", root)
    run(
        [
            uv,
            "run",
            "python",
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            "jigsaw-rules",
            "--display-name",
            "Python (Jigsaw Rules)",
        ],
        "notebook_kernel",
        root,
    )
    run([uv, "run", "python", "scripts/verify.py"], "verification", root)
    run([uv, "run", "jigsaw", "preflight"], "preflight", root)
    print("BOOTSTRAP_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
