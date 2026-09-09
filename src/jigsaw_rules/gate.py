"""Read-only feature-research completion gate, separate from reference reproduction."""

from __future__ import annotations

from pathlib import Path

from jigsaw_rules.diagnostics import diagnostic_evidence
from jigsaw_rules.expanded import expanded_evidence
from jigsaw_rules.instructions import instruction_evidence
from jigsaw_rules.pairs import pairs_evidence
from jigsaw_rules.released import released_evidence
from jigsaw_rules.research import research_evidence
from jigsaw_rules.resolution import resolution_evidence
from jigsaw_rules.retrieval import retrieval_evidence
from jigsaw_rules.review import public_evidence
from jigsaw_rules.robustness import robustness_evidence


def feature_gate(root: Path) -> dict:
    baseline = public_evidence(root, "baseline")
    released = released_evidence(root)
    evidence = {
        "research": research_evidence(root),
        "sensitivity": diagnostic_evidence(root),
        "pairs": pairs_evidence(root),
        "robustness": robustness_evidence(root),
        "instructions": instruction_evidence(root),
        "expanded": expanded_evidence(root),
        "retrieval": retrieval_evidence(root),
        "resolution": resolution_evidence(root),
    }
    missing = [name for name, item in evidence.items() if item is None]
    rules = len(baseline["audit"]["by_rule"])
    return {
        "schema": 2,
        "status": "open",
        "final_training_authorized_by_evidence": False,
        "labeled_rules": evidence["expanded"]["audit"]["development_policies"]
        if evidence["expanded"]
        else rules,
        "training_rows": evidence["expanded"]["audit"]["development_rows"]
        if evidence["expanded"]
        else baseline["training_rows"],
        "historical_reference": {
            "labeled_rules": rules,
            "training_rows": baseline["training_rows"],
        },
        "expanded_development": {
            "rows": evidence["expanded"]["audit"]["development_rows"]
            if evidence["expanded"]
            else 0,
            "policies": evidence["expanded"]["audit"]["development_policies"]
            if evidence["expanded"]
            else 0,
            "fitted_models": evidence["expanded"]["audit"]["actual_fitted_models"]
            if evidence["expanded"]
            else 0,
        },
        "released_data": {
            "state": "boundary verified; confirmation reserved" if released else "not prepared",
            "rules_available": released["boundary"]["rule_count"] if released else 0,
            "additional_research_rows": released["boundary"]["role_counts"]["research"]
            if released
            else 0,
            "reserved_rows": released["boundary"]["role_counts"]["confirmation"] if released else 0,
            "new_study_executed": evidence["expanded"] is not None,
        },
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
                "requirement": "Four-policy feature search and conflict/support sensitivities",
                "state": "verified" if evidence["expanded"] else "in progress",
            },
            {
                "requirement": "Policy-excluded target-derived semantic retrieval",
                "state": "verified" if evidence["retrieval"] else "in progress",
            },
            {
                "requirement": "Matryoshka embedding-resolution sensitivity",
                "state": "verified" if evidence["resolution"] else "in progress",
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
        "are exploratory; development comparisons are not independent confirmation. "
        "The four-policy study preserves financial advice, spoilers and the former private split. "
        "Stable feature selection, a stopping rationale and locked confirmation are still needed. "
        "Feature count and successful execution do not close this gate.",
    }


def require_feature_completion(root: Path) -> None:
    status = feature_gate(root)
    if not status["final_training_authorized_by_evidence"]:
        raise ValueError("Feature research gate is open: " + status["decision"])
