"""Run the project quality gate with live child output and UTC heartbeats."""

import subprocess
import sys
from pathlib import Path

from jigsaw_rules.runtime import Progress


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    commands = [
        [sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"],
        [sys.executable, "-m", "ruff", "check", "."],
        [sys.executable, "-m", "ruff", "format", "--check", "."],
        [sys.executable, "-m", "pytest", "-q"],
        [sys.executable, "scripts/build_notebooks.py", "--check"],
    ]
    with Progress(root / "logs/quality.jsonl", "quality_gate") as log:
        for command in commands:
            log.emit("check_started", command=command[2:])
            subprocess.run(command, cwd=root, check=True, timeout=600)
        log.emit("QUALITY_GATE_PASSED")


if __name__ == "__main__":
    main()
