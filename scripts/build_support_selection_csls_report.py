"""Render the completed CSLS support-selection diagnostic from public aggregates only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

POLICIES = {0: "Advertising", 1: "Legal advice"}
EXPECTED_JOB = "jigsaw-csls-support-audit-20260911-043145"


def evidence(root: Path) -> dict:
    folder = Path(root) / "reports/support_selection_csls"
    meta = json.loads((folder / "metadata.json").read_text())
    if meta.get("processing_job") != EXPECTED_JOB or meta.get("decision") != "STOP_CSLS_SELECTOR":
        raise ValueError("Unexpected CSLS evidence identity")
    expected = meta.get("files", {})
    if set(expected) != {"audit.json"}:
        raise ValueError("Unexpected CSLS evidence files")
    path = folder / "audit.json"
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected["audit.json"]:
        raise ValueError("CSLS audit checksum differs")
    audit = json.loads(path.read_text())
    if audit.get("query_targets_read") is not False or audit.get("prediction_arrays_read") is not False:
        raise ValueError("Target-free CSLS boundary differs")
    if audit.get("model_calls") != 0 or audit.get("gpu") is not False:
        raise ValueError("CSLS diagnostic unexpectedly used model compute")
    if audit.get("all_policies_eligible_for_blinded_relevance_review") is not False:
        raise ValueError("CSLS stopping decision differs")
    if [f["queries"] for f in audit["folds"]] != [234, 647]:
        raise ValueError("Unexpected CSLS cohort")
    if any(f["gate"]["eligible_for_blinded_relevance_review"] for f in audit["folds"]):
        raise ValueError("A policy unexpectedly passed the CSLS review gate")
    return audit


def rows(audit: dict) -> list[dict]:
    out = []
    for fold in audit["folds"]:
        out.append(
            {
                "Policy": POLICIES[fold["fold"]],
                "Queries": fold["queries"],
                "Changed pairs": fold["changed_pairs"],
                "Changed pairs (%)": round(100 * fold["changed_pair_fraction"], 2),
                "Raw positive max reuse (%)": round(
                    100 * fold["raw"]["positive"]["maximum_reuse_fraction"], 2
                ),
                "CSLS positive max reuse (%)": round(
                    100 * fold["csls"]["positive"]["maximum_reuse_fraction"], 2
                ),
                "Raw negative max reuse (%)": round(
                    100 * fold["raw"]["negative"]["maximum_reuse_fraction"], 2
                ),
                "CSLS negative max reuse (%)": round(
                    100 * fold["csls"]["negative"]["maximum_reuse_fraction"], 2
                ),
                "Gate passed": fold["gate"]["eligible_for_blinded_relevance_review"],
            }
        )
    return out


def notebook_cells() -> list[tuple[str, str]]:
    return [
        (
            "md",
            "### Follow-up diagnostic: hubness-corrected semantic selection\n\n"
            "**Decision: stop the CSLS selector before review or GPU inference.** The raw semantic audit suggested hub concentration, so this bounded CPU-only follow-up applied CSLS-style local scaling to the same cached adapted vectors. No query targets, saved predictions, new embeddings, model calls, GPU, training, or new AUC were used.\n\n"
            "CSLS changed 114/234 advertising pairs and 268/647 legal-advice pairs. It increased the number of distinct supports, but failed the predeclared requirement that maximum reuse strictly improve for both classes in every policy. Most importantly, legal-advice negative-example maximum reuse worsened from **18.70% to 19.63%**. That failure ends this candidate.",
        ),
        (
            "code",
            "from scripts.build_support_selection_csls_report import evidence as csls_evidence, rows as csls_rows\n"
            "csls = csls_evidence(root)\n"
            "display(pd.DataFrame(csls_rows(csls)))\n"
            "print('All-policy review gate:', csls['all_policies_eligible_for_blinded_relevance_review'])\n"
            "print('GPU inference authorized:', csls['gpu_inference_authorized'])\n"
            "print('No AUC was calculated in this target-free diagnostic.')",
        ),
        (
            "md",
            "The next feature-engineering priority is **training-signal quality**, not another nearest-neighbor variant. The current adaptation builder drops conflicting normalized `(rule, body)` labels entirely. The published protocol recorded 2 conflicting pairs / 79 conflicting occurrences for advertising and 10 / 339 for legal advice. Leading competition systems also treated duplicate/conflicting supervision explicitly. The next milestone therefore starts with a CPU-only provenance/vote audit before any retraining.",
        ),
    ]
