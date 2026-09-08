"""Recompute saved feature-study metrics from private OOF files without fitting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from jigsaw_rules.data import load_data
from jigsaw_rules.diagnostics import diagnostic_evidence
from jigsaw_rules.instructions import instruction_evidence
from jigsaw_rules.metrics import evaluate
from jigsaw_rules.pairs import pairs_evidence
from jigsaw_rules.research import research_evidence, verify_research
from jigsaw_rules.robustness import robustness_evidence
from jigsaw_rules.runtime import Progress, digest


def compare_metrics(actual, expected):
    if set(actual) != set(expected):
        raise ValueError("Saved metric schema differs")
    for name, value in expected.items():
        if isinstance(value, dict):
            compare_metrics(actual[name], value)
        elif isinstance(value, (int, float)):
            if not np.isclose(actual[name], value, rtol=1e-10, atol=1e-10):
                raise ValueError(f"Saved metric differs: {name}")
        elif actual[name] != value:
            raise ValueError(f"Saved metric differs: {name}")


def verify_predictions(train, oof, records):
    verified = 0
    for record in records:
        if record["model"] not in set(oof.model):
            # Reference results are verified by the broad-study verifier.
            if record["model"] != "original_reference":
                raise ValueError("Missing model predictions")
            continue
        part = oof[(oof.model == record["model"]) & (oof.protocol == record["protocol"])]
        if part.row_id.duplicated().any() or set(part.row_id) != set(train.row_id):
            raise ValueError("OOF coverage differs")
        part = part.set_index("row_id").loc[train.row_id]
        if not np.array_equal(part.rule_violation, train.rule_violation) or not np.array_equal(
            part.rule, train.rule
        ):
            raise ValueError("OOF target or rule identity differs")
        compare_metrics(
            evaluate(train.rule_violation, part.probability, train.rule), record["metrics"]
        )
        verified += 1
    return verified


def main():
    root = Path(__file__).resolve().parents[1]
    train, _, _ = load_data(root / "data/raw")
    with Progress(root / "logs/research_verification.jsonl", "private_feature_verification") as log:
        broad = research_evidence(root)
        if broad is None:
            raise ValueError("Missing broad research")
        verify_research(root, root / "runs" / broad["metadata"]["run_id"])
        for name, reader in (
            ("sensitivity", diagnostic_evidence),
            ("pairs", pairs_evidence),
            ("robustness", robustness_evidence),
            ("instructions", instruction_evidence),
        ):
            evidence = reader(root)
            if evidence is None:
                if name == "instructions":
                    log.emit("PENDING_INSTRUCTION_STUDY")
                    continue
                raise ValueError(f"Missing {name} evidence")
            path = root / "runs" / evidence["metadata"]["run_id"] / "review"
            marker = json.loads((path / "complete.json").read_text())
            for filename, expected in marker["files"].items():
                if digest(path / filename) != expected:
                    raise ValueError(f"Private {name} review checksum differs")
            for filename in evidence["metadata"]["files"]:
                if (path / filename).read_bytes() != (
                    root / "reports" / name / filename
                ).read_bytes():
                    raise ValueError(f"Public {name} export differs from private review")
            count = verify_predictions(train, pd.read_csv(path / "oof.csv"), evidence["results"])
            log.emit("PRIVATE_FEATURE_METRICS_VERIFIED", study=name, metric_records=count)
        log.emit("ALL_FEATURE_OOF_METRICS_VERIFIED", fitted_models=0)


if __name__ == "__main__":
    main()
