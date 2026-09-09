"""Publish only aggregates from the verified target-blind reserve audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from jigsaw_rules.embeddings import content_key
from jigsaw_rules.final_model import verify_stage
from jigsaw_rules.runtime import atomic_bytes, atomic_json, digest

ROOT = Path(__file__).resolve().parents[1]


def build(root: Path, run_id: str) -> None:
    if len(run_id) != 20 or any(c not in "0123456789abcdef" for c in run_id):
        raise ValueError("Expected a content-addressed eligibility run")
    path = root / "runs/confirmation_inputs" / run_id / "eligibility"
    verify_stage(path)
    provenance = json.loads((path / "provenance.json").read_text())
    if content_key(provenance)[:20] != run_id or any(
        digest(root / "src/jigsaw_rules" / name) != sha
        for name, sha in provenance["implementation"].items()
    ):
        raise ValueError("Eligibility source or run identity differs")
    table = pd.read_csv(path / "eligibility.csv")
    ids = json.loads((path / "eligible_ids.json").read_text())
    audit = json.loads((path / "audit.json").read_text())
    if (
        table.row_id.duplicated().any()
        or ids != table.loc[table.eligible, "row_id"].tolist()
        or audit["eligible_ids_sha256"] != content_key(ids)
        or audit["eligible_rows"] != len(ids)
        or audit["reserved_rows"] != len(table)
        or audit["excluded_near_copy_rows"] != int(table.near_development_copy.sum())
        or audit["confirmation_targets_accessed"]
    ):
        raise ValueError("Eligibility aggregates differ from private identities")
    folder = root / "reports/confirmation_inputs"
    atomic_bytes(folder / "audit.json", (path / "audit.json").read_bytes())
    atomic_json(
        folder / "metadata.json",
        {
            "run_id": run_id,
            "provenance": provenance,
            "audit_sha256": digest(path / "audit.json"),
            "private_checkpoint_sha256": digest(path / "complete.json"),
            "exporter_sha256": digest(root / "scripts/build_confirmation_report.py"),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    build(ROOT, parser.parse_args().run_id)
