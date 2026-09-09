"""Execute the standalone notebook on pinned original training and preview inputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from jigsaw_rules.data import load_data
from jigsaw_rules.released import download_release, load_plan
from jigsaw_rules.runtime import atomic_bytes, atomic_json, digest


def prepare_original(root: Path) -> dict:
    plan = load_plan(root)
    source = download_release(root, plan)
    mapping = {
        "train.csv": "train.csv",
        "test.csv": "public_test.csv",
        "sample_submission.csv": "public_sample_submission.csv",
    }
    destination = root / "data/raw"
    payloads = {}
    for name, original in mapping.items():
        path = destination / name
        expected = plan["source"]["files"][original]["sha256"]
        if path.exists() and digest(path) != expected:
            raise ValueError("Existing original input differs; no file was replaced")
        payloads[path] = (source / original).read_bytes()
    for path, data in payloads.items():
        if not path.exists():
            atomic_bytes(path, data)
    train, preview, _ = load_data(destination)
    return {
        "training_rows": len(train),
        "preview_rows": len(preview),
        "inputs": {name: digest(destination / name) for name in mapping},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    record = prepare_original(root)
    if args.prepare_only:
        print(json.dumps(record))
        return
    command = [sys.executable, "scripts/execute_notebooks.py", "--kaggle"]
    subprocess.run(command, cwd=root, check=True, timeout=180)
    first = digest(root / "kaggle_output/submission.csv")
    subprocess.run(command, cwd=root, check=True, timeout=180)
    if digest(root / "kaggle_output/submission.csv") != first:
        raise ValueError("Original-data notebook replay changed the submission")
    manifest = json.loads((root / "kaggle_output/submission_manifest.json").read_text())
    if manifest["synthetic"] or manifest["rows"] != record["preview_rows"]:
        raise ValueError("Wrong submission validation data")
    atomic_json(
        root / "reports/checkpoints/submission.json",
        {
            "schema": 1,
            "status": "passed",
            **record,
            "model": manifest["model"],
            "notebook": "kaggle/reference.ipynb",
            "notebook_sha256": digest(root / "kaggle/reference.ipynb"),
            "source_sha256": manifest["source_sha256"],
            "submission_sha256": first,
            "jupyter_executed": True,
            "replay_passed": True,
            "scope": "Historical lexical preview; not verification of the neural submission",
        },
    )
    print("ORIGINAL_SUBMISSION_NOTEBOOK_VERIFIED")


if __name__ == "__main__":
    main()
