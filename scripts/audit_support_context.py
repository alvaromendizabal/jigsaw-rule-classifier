"""Audit support selection diversity without model inference or query targets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from jigsaw_rules.runtime import atomic_json, digest
from scripts.competition_features import content_hash
from scripts.support_context import retrieve


def run(root, plan_path, output):
    spec = json.loads((root / "configs/support_context.json").read_text())
    if digest(plan_path) != spec["plan_sha256"]:
        raise ValueError("Original target-free plan differs")
    plan = json.loads(plan_path.read_text())
    audit = {
        "plan_sha256": digest(plan_path),
        "audit_source_sha256": digest(Path(__file__)),
        "retrieval_source_sha256": digest(root / "scripts/support_context.py"),
        "query_targets_read": False,
        "meaning": "Label-free selection diversity; neither relevance nor predictive value",
        "folds": [],
    }
    for fold in plan["folds"]:
        pairs = retrieve(fold)
        classes = {}
        for name in ("violating_examples", "permitted_examples"):
            counts = Counter(pair[name][0] for pair in pairs)
            classes[name] = {
                "unique_selected": len(counts),
                "max_query_share": max(counts.values()) / len(pairs),
            }
        audit["folds"].append(
            {
                "rule": fold["rule"],
                "queries": len(pairs),
                "support_selection_sha256": content_hash(pairs),
                "class_selection": classes,
            }
        )
    atomic_json(output, audit)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(Path(__file__).resolve().parents[1], args.plan, args.output)))
