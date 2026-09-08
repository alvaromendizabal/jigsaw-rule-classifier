"""Read-only feature-research completion gate, separate from reference reproduction."""

from __future__ import annotations

from pathlib import Path

from jigsaw_rules.diagnostics import diagnostic_evidence
from jigsaw_rules.instructions import instruction_evidence
from jigsaw_rules.pairs import pairs_evidence
from jigsaw_rules.research import research_evidence
from jigsaw_rules.review import public_evidence
from jigsaw_rules.robustness import robustness_evidence


def feature_gate(root: Path) -> dict:
    baseline = public_evidence(root, "baseline")
    evidence = {
        "research": research_evidence(root),
        "sensitivity": diagnostic_evidence(root),
        "pairs": pairs_evidence(root),
        "robustness": robustness_evidence(root),
        "instructions": instruction_evidence(root),
    }
    missing = [name for name, item in evidence.items() if item is None]
    rules = len(baseline["audit"]["by_rule"])
    return {
        "schema": 1,
        "status": "open",
        "final_training_authorized_by_evidence": False,
        "labeled_rules": rules,
        "training_rows": baseline["training_rows"],
        "studies": {name: item["metadata"]["run_id"] for name, item in evidence.items() if item},
        "missing_studies": missing,
        "criteria": [
            {
                "requirement": "Broad candidate generation and training-only screening",
                "state": "verified" if evidence["research"] else "missing",
            },
            {
                "requirement": "Matched additions, removals, permutation and stability",
                "state": "verified" if evidence["research"] else "missing",
            },
            {
                "requirement": "Unscreened controls and support-geometry alternatives",
                "state": "verified" if evidence["sensitivity"] else "missing",
            },
            {
                "requirement": "Joint comment-rule and support entailment representations",
                "state": "verified" if evidence["pairs"] else "missing",
            },
            {
                "requirement": "Self-support and approximate-copy stress tests",
                "state": "verified"
                if evidence["robustness"] and evidence["sensitivity"]
                else "missing",
            },
            {
                "requirement": "Fixed instruction-likelihood and context ablations",
                "state": "verified" if evidence["instructions"] else "in progress",
            },
            {
                "requirement": "Independent confirmation after feature-family selection",
                "state": "not established",
            },
            {
                "requirement": "Representative coverage of unseen policy types",
                "state": "not established",
            },
            {
                "requirement": "Selected final representation and probability-quality acceptance",
                "state": "not established",
            },
            {
                "requirement": "Research candidate promoted into offline inference",
                "state": "blocked until prior requirements pass",
            },
        ],
        "decision": "Retain the reproducible lexical reference. Candidate OOF comparisons "
        "are exploratory; repeated selection on two policies is not independent confirmation. "
        "Feature count and successful execution do not close this gate.",
    }


def require_feature_completion(root: Path) -> None:
    status = feature_gate(root)
    if not status["final_training_authorized_by_evidence"]:
        raise ValueError("Feature research gate is open: " + status["decision"])
