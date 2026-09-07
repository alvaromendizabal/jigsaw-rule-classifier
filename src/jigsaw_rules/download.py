"""Use official Kaggle authentication and independently commit the three CSVs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from filelock import FileLock

from jigsaw_rules.data import FILES, load_data
from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest

COMPETITION = "jigsaw-agile-community-rules"


def download(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    executable = shutil.which("kaggle")
    if executable is None:
        raise RuntimeError("Kaggle CLI is missing; run the project bootstrap")
    with FileLock(str(directory / "download.lock"), timeout=1):
        with Progress(directory / "download.jsonl", "download") as log:
            manifest_path = directory / "download_manifest.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            for name in FILES:
                dest = directory / name
                if dest.exists() and manifest.get(name) == digest(dest):
                    log.emit("reused", file=name)
                    continue
                if dest.exists() and name not in manifest:
                    log.emit("adopting_existing_file", file=name)
                else:
                    staging = directory / ".downloads"
                    staging.mkdir(exist_ok=True)
                    command = [
                        executable,
                        "competitions",
                        "download",
                        "-c",
                        COMPETITION,
                        "-f",
                        name,
                        "-p",
                        str(staging),
                        "--force",
                    ]
                    subprocess.run(command, check=True, timeout=600, env=os.environ.copy())
                    csv_path, zip_path = staging / name, staging / f"{name}.zip"
                    if csv_path.is_file():
                        payload = csv_path.read_bytes()
                    elif zip_path.is_file():
                        with zipfile.ZipFile(zip_path) as archive:
                            if (
                                archive.namelist() != [name]
                                or archive.getinfo(name).file_size > 100_000_000
                            ):
                                raise ValueError("Unexpected competition file archive")
                            payload = archive.read(name)
                    else:
                        raise FileNotFoundError(f"Kaggle did not create {name}")
                    atomic_bytes(dest, payload)
                manifest[name] = digest(dest)
                atomic_json(manifest_path, manifest)
            load_data(directory)
            log.emit("schema_validated", files=3)
