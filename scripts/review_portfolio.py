"""Verify original aggregate receipts; this is not a hidden-label evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def review(root: Path) -> dict:
    root = Path(root).resolve()
    hashes = {}
    for section in ("baseline", "support_adaptation"):
        folder = root / "reports" / section
        metadata = json.loads((folder / "metadata.json").read_text())
        for name, expected in metadata["files"].items():
            path = (folder / name).resolve()
            require(path.is_relative_to(folder), "Evidence path escapes its section")
            require(path.is_file(), "Missing evidence: " + name)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            require(digest == expected, "Evidence checksum mismatch: " + name)
            hashes[path.relative_to(root).as_posix()] = digest
    receipt_path = root / "reports/checkpoints/kaggle_adaptation.json"
    receipt = json.loads(receipt_path.read_text())
    require(receipt["status"] == "submitted_and_scored", "Unverified submission state")
    submission = receipt["submission"]
    require(submission["status"] == "succeeded", "Submission did not succeed")
    require(submission["public_score"] == 0.91808, "Public result changed")
    require(submission["private_score"] == 0.91425, "Private result changed")
    comparison = receipt["comparison"]
    require(
        math.isclose(
            submission["private_score"] - comparison["baseline_private_auc"],
            comparison["private_auc_gain"],
            abs_tol=1e-12,
        ),
        "Incorrect baseline comparison",
    )
    folder = root / "reports/support_adaptation"
    protocol = json.loads((folder / "protocol.json").read_text())
    require(protocol["rows"] == 881 and protocol["policies"] == 2, "Wrong comparison population")
    require(
        sum(row["novel_query_rows"] for row in protocol["folds"]) == 881,
        "Query accounting mismatch",
    )
    results = json.loads((folder / "results.json").read_text())
    models = {row["model"]: row for row in results}
    scores = {}
    for name in ("frozen_raw", "adapted_raw"):
        metrics = models[name]["metrics"]
        values = list(metrics["per_rule_auc"].values())
        require(
            len(values) == 2 and all(math.isfinite(v) and 0 <= v <= 1 for v in values),
            "Invalid policy AUC",
        )
        require(
            math.isclose(sum(values) / 2, metrics["rule_macro_auc"], abs_tol=1e-12),
            "Macro AUC must weight policies equally",
        )
        scores[name] = metrics["rule_macro_auc"]
    require(
        set(models["frozen_raw"]["metrics"]["per_rule_auc"])
        == set(models["adapted_raw"]["metrics"]["per_rule_auc"]),
        "Mismatched policy cohorts",
    )
    hashes[receipt_path.relative_to(root).as_posix()] = hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()
    return {
        "status": "passed",
        "scope": "Original aggregate arithmetic and integrity; no hidden labels or retraining",
        "historical_submission": {
            "public_auc": submission["public_score"],
            "private_auc": submission["private_score"],
        },
        "matched_development": {
            "rows": 881,
            "policies": 2,
            **scores,
            "gain": scores["adapted_raw"] - scores["frozen_raw"],
        },
        "source_sha256": hashes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    print(json.dumps(review(args.root), indent=2))


if __name__ == "__main__":
    main()
