"""Resolve the official competition mount across Kaggle runtime layouts."""

import os
from pathlib import Path


def submission_input_root(local_root: Path, kaggle_root: Path = Path("/kaggle/input")) -> Path:
    override = os.environ.get("JIGSAW_KAGGLE_INPUT")
    if override:
        return Path(override)
    if not kaggle_root.is_dir():
        return local_root / "data/raw"
    slug = "jigsaw-agile-community-rules"
    for directory in (kaggle_root / "competitions" / slug, kaggle_root / slug):
        if all(
            (directory / name).is_file()
            for name in ("train.csv", "test.csv", "sample_submission.csv")
        ):
            return directory
    raise FileNotFoundError(
        "Attach the official Jigsaw - Agile Community Rules Classification competition "
        "data in Kaggle's Input panel, then Save Version and Run All again."
    )
