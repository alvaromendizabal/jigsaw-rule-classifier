"""User-run offline inference with durable model/batch checkpoints and verified downloads."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd
from filelock import FileLock

from jigsaw_rules.data import FILES, load_data, validate_submission
from jigsaw_rules.model import LexicalClassifier
from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest, stage


def generate_submission(
    input_root: Path,
    output_root: Path,
    cache_root: Path,
    *,
    source_sha256: str,
    batch_size: int = 5000,
    seed: int = 2025,
) -> Path:
    """Fit once, resume completed prediction batches, validate, then publish locally.

    Only call this for the user's explicit generation action or synthetic software tests.
    It never calls the Kaggle API. Cache files are trusted locally created artifacts.
    """
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
        or not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(c not in "0123456789abcdef" for c in source_sha256)
    ):
        raise ValueError("Positive batch_size and a full source SHA-256 are required")
    input_root, output_root, cache_root = map(Path, (input_root, output_root, cache_root))
    inputs = {name: digest(input_root / name) for name in FILES}
    train, test, sample = load_data(input_root)
    contract = {
        "schema": 1,
        "source_sha256": source_sha256,
        "inputs": inputs,
        "synthetic": (input_root / "SYNTHETIC.txt").exists(),
        "model": "rule_examples",
        "seed": seed,
        "batch_size": batch_size,
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "joblib")},
    }
    key = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    directory = cache_root / key
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output_root / "submission.lock"), timeout=1):
        with FileLock(str(cache_root / f"{key}.lock"), timeout=1):
            with Progress(directory / "events.jsonl", "offline_submission") as log:

                def unchanged() -> None:
                    if inputs != {name: digest(input_root / name) for name in FILES}:
                        raise ValueError("Input files changed during generation; export refused")

                unchanged()
                atomic_json(directory / "contract.json", contract)
                log.emit("data_validated", train_rows=len(train), test_rows=len(test))

                def fit(destination: Path) -> None:
                    fitted = LexicalClassifier(context=True, seed=seed).fit(train)
                    joblib.dump(fitted, destination / "model.joblib")

                fitted_path = stage(directory, "model", fit)
                fitted = None
                chunks = []
                for offset in range(0, len(test), batch_size):
                    end = min(offset + batch_size, len(test))

                    def predict(destination: Path, start=offset, stop=end) -> None:
                        nonlocal fitted
                        if fitted is None:
                            fitted = joblib.load(fitted_path / "model.joblib")
                        frame = pd.DataFrame(
                            {
                                "row_id": test.iloc[start:stop].row_id.to_numpy(),
                                "rule_violation": fitted.predict(test.iloc[start:stop]),
                            }
                        )
                        validate_submission(frame, sample.iloc[start:stop].reset_index(drop=True))
                        atomic_bytes(
                            destination / "predictions.csv", frame.to_csv(index=False).encode()
                        )

                    part = stage(directory, f"batch_{offset:09d}", predict)
                    frame = pd.read_csv(part / "predictions.csv")
                    validate_submission(frame, sample.iloc[offset:end].reset_index(drop=True))
                    chunks.append(frame)
                    log.emit("prediction_batch", completed_rows=end, total_rows=len(test))
                submission = pd.concat(chunks, ignore_index=True)
                validate_submission(submission, sample)
                unchanged()
                payload = submission.to_csv(index=False).encode()
                manifest = {
                    **contract,
                    "contract_sha256": key,
                    "rows": len(submission),
                    "submission_sha256": hashlib.sha256(payload).hexdigest(),
                    "status": "Validated local inference; no Kaggle score or upload.",
                }
                atomic_bytes(output_root / "submission.csv", payload)
                atomic_json(output_root / "submission_manifest.json", manifest)
                log.emit("SUBMISSION_VALIDATED", rows=len(submission), contract=key)
    return output_root / "submission.csv"


def download_link(path: Path, *, max_bytes: int = 10_000_000) -> str:
    """Return a click-to-download HTML link only for a checksum-verified generated CSV.

    A small data URI avoids fragile Jupyter/SageMaker proxy-relative URLs. Large files
    use the notebook file browser rather than embedding an unbounded payload.
    """
    path = Path(path)
    manifest = json.loads((path.parent / "submission_manifest.json").read_text())
    if path.name != "submission.csv" or digest(path) != manifest.get("submission_sha256"):
        raise ValueError("Submission checksum mismatch; regenerate before downloading")
    if path.stat().st_size > max_bytes:
        return (
            "<p>Validated file is large. Download submission.csv from the output file browser.</p>"
        )
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["submission_sha256"]:
        raise ValueError("Submission changed while creating the download link")
    data = base64.b64encode(payload).decode("ascii")
    name = html.escape(path.name, quote=True)
    return f'<a download="{name}" href="data:text/csv;base64,{data}">Download {name}</a>'
