"""Convert manually recorded blind A/B relevance ratings into private-safe counts.

No query targets or comment bodies are opened. This is not an AUC evaluator and
cannot authorize inference. Ratings must cover the frozen sample exactly once.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.audit_support_selection import FINAL_FILES
from scripts.support_selection_runtime import (
    atomic_json,
    digest,
    json_bytes,
    load_stage,
    save_stage,
    sha_bytes,
)


def summarize(output: Path, ratings_path: Path) -> dict:
    current = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    run_id = current["run_id"]
    if len(run_id) != 24 or any(c not in "0123456789abcdef" for c in run_id):
        raise ValueError("Invalid run identity")
    run = output / run_id
    if load_stage(run, run_id, FINAL_FILES) is None:
        raise ValueError("Audit is not complete")
    supplied = json.loads(ratings_path.read_text(encoding="utf-8"))
    if set(supplied) != {
        "schema",
        "run_id",
        "query_targets_read",
        "contains_comment_text",
        "ratings",
    }:
        raise ValueError("Unexpected rating schema")
    if (
        supplied["schema"] != 1
        or supplied["run_id"] != run_id
        or supplied["query_targets_read"] is not False
        or supplied["contains_comment_text"] is not False
    ):
        raise ValueError("Rating identity or data boundary differs")
    key = json.loads((run / "private/qualitative_key.json").read_text(encoding="utf-8"))
    by_id = {r["case_id"]: r for r in key}
    seen, policies = set(), {}
    allowed_issues = {
        "none",
        "request_offer",
        "quotation",
        "negation",
        "exception",
        "topic_not_behavior",
        "other_unclear",
    }
    for row in supplied["ratings"]:
        if set(row) != {"case_id", "fold", "preference", "issue"}:
            raise ValueError("Unexpected per-case rating fields")
        case = row["case_id"]
        if case not in by_id or case in seen or row["fold"] != by_id[case]["fold"]:
            raise ValueError("Unknown, duplicate, or mismatched review case")
        if (
            row["preference"] not in {"A", "B", "tie", "unclear"}
            or row["issue"] not in allowed_issues
        ):
            raise ValueError("Review is incomplete or contains unrecognized values")
        seen.add(case)
        policy = policies.setdefault(
            str(row["fold"]), {"preferences": Counter(), "issues": Counter()}
        )
        choice = (
            by_id[case][row["preference"]] if row["preference"] in {"A", "B"} else row["preference"]
        )
        policy["preferences"][choice] += 1
        policy["issues"][row["issue"]] += 1
    if seen != set(by_id):
        raise ValueError("All predeclared cases must be reviewed; no cherry-picking")
    result = {
        "schema": 1,
        "run_id": run_id,
        "status": "blinded_relevance_review_recorded",
        "cases": len(seen),
        "policies": policies,
        "ratings_sha256": digest(ratings_path),
        "query_targets_read": False,
        "contains_comment_text": False,
        "official_metric": None,
        "automatic_inference_allowed": False,
        "limitation": "One subjective relevance review is not target labeling or efficacy evidence.",
    }
    destination = run / "review"
    identity = sha_bytes(json_bytes({"run": run_id, "ratings": result["ratings_sha256"]}))
    old = load_stage(destination, identity, {"review_summary.json"})
    if old is None:
        save_stage(destination, identity, {"review_summary.json": json_bytes(result)})
    atomic_json(output / "review_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/support_selection"))
    parser.add_argument("--ratings", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(summarize(args.output, args.ratings), indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Review stopped: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
