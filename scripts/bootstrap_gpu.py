"""Isolate Python dependencies while reusing the image's immutable CUDA packages."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import venv
from importlib.metadata import distributions
from pathlib import Path


def install_pip(destination: Path, wheel: Path) -> None:
    """Seed an isolated venv from a verified wheel when ensurepip is unavailable."""
    environment = dict(os.environ, PYTHONPATH=str(wheel.resolve()))
    subprocess.run(
        [
            str(destination / "bin/python"),
            "-m",
            "pip",
            "install",
            "--ignore-installed",
            "--no-deps",
            "--no-index",
            str(wheel.resolve()),
        ],
        env=environment,
        check=True,
        timeout=60,
    )


def bootstrap(destination: Path, pip_wheel: Path | None = None) -> None:
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an existing GPU environment")
    native = []
    for distribution in distributions():
        name = distribution.metadata["Name"].lower().replace("_", "-")
        if name in {"torch", "triton"} or name.startswith(("nvidia-", "cuda-")):
            native.append(distribution)
    if not any(d.metadata["Name"].lower() == "torch" for d in native):
        raise RuntimeError("The pinned training image must supply PyTorch")
    venv.EnvBuilder(with_pip=pip_wheel is None, system_site_packages=False).create(destination)
    target = (
        destination / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    )
    copied = {}
    for distribution in native:
        for name in sorted({p.parts[0] for p in distribution.files or []}):
            if name in {"..", "."}:
                continue
            source = Path(distribution.locate_file(name)).resolve()
            if not source.exists():
                continue
            output = target / name
            if output.exists():
                if copied.get(name) != source:
                    raise ValueError("Native package paths collide: " + name)
            else:
                # Writable package links let installers corrupt the parent runtime.
                # Private copies plus exact constraints preserve that runtime.
                if source.is_dir():
                    shutil.copytree(source, output)
                else:
                    shutil.copy2(source, output)
                copied[name] = source
    (destination / "native-constraints.txt").write_text(
        "\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in native)) + "\n"
    )
    if pip_wheel is not None:
        install_pip(destination, pip_wheel)
    print(
        "Isolated GPU environment; reused native packages:",
        sorted(d.metadata["Name"] for d in native),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--pip-wheel", type=Path)
    args = parser.parse_args()
    bootstrap(args.destination, args.pip_wheel)
