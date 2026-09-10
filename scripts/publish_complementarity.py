"""Export the completed study's aggregate JSON; private predictions stay in runs/S3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jigsaw_rules.runtime import atomic_bytes, atomic_json, digest

AGGREGATES = (
    "results.json",
    "uncertainty.json",
    "decision.json",
    "correlations.json",
    "protocol.json",
    "provenance.json",
    "inference.json",
)


def publish(
    root: Path, evaluation: Path, preregistration_commit: str, study: str = "complementarity"
) -> Path:
    if study not in {"complementarity", "backbone_capacity"}:
        raise ValueError("Unknown registered model study")
    marker = json.loads((evaluation / "complete.json").read_text())
    for name, sha in marker["files"].items():
        path = evaluation / name
        if evaluation.resolve() not in path.resolve().parents or digest(path) != sha:
            raise ValueError("Completed evaluation checksum differs")
    if not set(AGGREGATES).issubset(marker["files"]):
        raise ValueError("Completed evaluation is missing aggregate evidence")
    provenance = json.loads((evaluation / "provenance.json").read_text())
    if provenance["spec"] != json.loads((root / "configs" / f"{study}.json").read_text()):
        raise ValueError("Evaluation configuration differs from registered study")
    if provenance["source"] != digest(root / "scripts" / f"evaluate_{study}.py"):
        raise ValueError("Evaluation source differs")
    destination = root / "reports" / study
    for name in AGGREGATES:
        atomic_bytes(destination / name, (evaluation / name).read_bytes())
    atomic_json(
        destination / "metadata.json",
        {
            "schema": 1,
            "study": "original_policy_model_" + study,
            "evaluation_run": evaluation.name,
            "code_preregistration_commit": preregistration_commit,
            "evaluated_utc": marker["finished_at"],
            "files": {name: digest(destination / name) for name in AGGREGATES},
        },
    )
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--preregistration-commit", required=True)
    parser.add_argument(
        "--study", choices=("complementarity", "backbone_capacity"), default="complementarity"
    )
    args = parser.parse_args()
    print(
        publish(
            Path(__file__).resolve().parents[1],
            args.evaluation,
            args.preregistration_commit,
            args.study,
        )
    )
