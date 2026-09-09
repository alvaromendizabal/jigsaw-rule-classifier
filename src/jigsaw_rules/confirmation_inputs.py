"""Target-blind reserve eligibility; this module never opens the solution CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from jigsaw_rules.data import EXAMPLES, normalize, validate_frame
from jigsaw_rules.embeddings import content_key
from jigsaw_rules.expanded import load_development
from jigsaw_rules.final_model import verify_stage
from jigsaw_rules.released import load_plan, released_evidence
from jigsaw_rules.robustness import POLICY, near_copy_edges
from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest, stage


def eligibility(training: pd.DataFrame, reserved: pd.DataFrame, log=None) -> pd.DataFrame:
    """Apply the existing label-free near-copy rule before confirmation target access."""
    validate_frame(training, train=False)
    validate_frame(reserved, train=False)
    exposure = sorted({normalize(t) for column in ["body", *EXAMPLES] for t in training[column]})
    bodies = reserved.body.map(normalize)
    if set(exposure) & set(bodies):
        raise ValueError("Reserved body was exposed to development text")
    unique = sorted(set(bodies))
    # Chunk both text lists to keep sparse similarity products bounded. The detector
    # has a fixed hashing vocabulary, so this is exactly the same pairwise rule.
    affected = set()
    for start in range(0, len(unique), 2048):
        block = unique[start : start + 2048]
        _, columns = near_copy_edges(exposure, block, POLICY)
        affected.update(block[i] for i in columns)
        if log:
            log.emit(
                "near_copy_block",
                reviewed=min(start + 2048, len(unique)),
                total=len(unique),
                affected_unique_bodies=len(affected),
            )
    output = reserved[["row_id", "rule"]].copy()
    output["near_development_copy"] = bodies.isin(affected).to_numpy()
    output["self_support_match"] = np.logical_or.reduce(
        [bodies.eq(reserved[column].map(normalize)).to_numpy() for column in EXAMPLES]
    )
    output["eligible"] = ~output.near_development_copy
    output["body_sha256"] = bodies.map(content_key).to_numpy()
    return output


def reserve_frame(root: Path) -> tuple[pd.DataFrame, dict]:
    plan, evidence = load_plan(root), released_evidence(root)
    if evidence is None:
        raise ValueError("Released boundary must be verified before reserve inputs")
    path = root / "data/released/v1/test.csv"
    if digest(path) != plan["source"]["files"]["test.csv"]["sha256"]:
        raise ValueError("Released target-free input differs")
    boundary = root / "runs/released" / evidence["metadata"]["run_id"] / "boundary"
    if digest(boundary / "complete.json") != evidence["metadata"]["private_checkpoint_sha256"]:
        raise ValueError("Released boundary identity differs")
    verify_stage(boundary)
    assignment = pd.read_csv(boundary / "assignments.csv")
    selected = assignment.loc[assignment.role.eq("confirmation")]
    frame = pd.read_csv(path)
    validate_frame(frame, train=False)
    if selected.row_id.duplicated().any() or not set(selected.row_id).issubset(frame.row_id):
        raise ValueError("Invalid reserved input identities")
    frame = frame.set_index("row_id").loc[selected.row_id].reset_index()
    if len(frame) != evidence["boundary"]["role_counts"]["confirmation"]:
        raise ValueError("Reserve row coverage differs")
    expected = selected.rule_id.map(plan["rule_text"]).to_numpy()
    if not np.array_equal(frame.rule, expected):
        raise ValueError("Reserved policy assignment differs")
    return frame, {
        "test_sha256": digest(path),
        "assignment_sha256": digest(boundary / "assignments.csv"),
        "boundary_run": evidence["metadata"]["run_id"],
    }


def prepare_confirmation(root: Path) -> Path:
    training = load_development(root).drop(columns="rule_violation")
    reserved, identity = reserve_frame(root)
    provenance = {
        **identity,
        "development_inputs_sha256": content_key(training.to_csv(index=False)),
        "implementation": {
            name: digest(root / "src/jigsaw_rules" / name)
            for name in ["confirmation_inputs.py", "robustness.py", "data.py"]
        },
        "near_copy_policy": POLICY,
        "primary_eligibility": (
            "Exclude reserved bodies approximately exposed in development; keep supplied "
            "self-support matches and report their removal as a fixed sensitivity"
        ),
        "targets_accessed": False,
    }
    directory = root / "runs/confirmation_inputs" / content_key(provenance)[:20]

    def action(path):
        with Progress(directory / "events.jsonl", "target_blind_eligibility") as log:
            table = eligibility(training, reserved, log)
        atomic_bytes(path / "eligibility.csv", table.to_csv(index=False).encode())
        ids = table.loc[table.eligible, "row_id"].tolist()
        atomic_json(path / "eligible_ids.json", ids)
        atomic_json(
            path / "audit.json",
            {
                "reserved_rows": len(table),
                "eligible_rows": len(ids),
                "excluded_near_copy_rows": int(table.near_development_copy.sum()),
                "eligible_self_support_rows": int(
                    (table.eligible & table.self_support_match).sum()
                ),
                "eligible_ids_sha256": content_key(ids),
                "by_policy": table.groupby("rule")
                .agg(
                    rows=("row_id", "size"),
                    eligible=("eligible", "sum"),
                    self_support=("self_support_match", "sum"),
                )
                .reset_index()
                .to_dict("records"),
                "confirmation_targets_accessed": False,
                "predictions_generated": False,
                "policy": POLICY,
                "limitation": (
                    "A conservative copy detector, not complete paraphrase or "
                    "conversation-origin isolation"
                ),
            },
        )
        atomic_json(path / "provenance.json", provenance)

    stage(directory, "eligibility", action)
    return directory
