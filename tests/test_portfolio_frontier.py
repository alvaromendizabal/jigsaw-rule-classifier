"""Regression guards for employer-facing frontier evidence."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "reports/checkpoints/cross_model_frontier_20261006.json"


def test_cross_model_receipt_is_public_safe_and_consistent() -> None:
    data = json.loads(RECEIPT.read_text())
    assert data["schema_version"] == 1
    assert data["metric"] == "policy_macro_roc_auc"
    assert data["evaluation"]["untouched_holdout"] is False
    assert data["evaluation"]["directly_comparable_to_official_score"] is False

    studies = {row["id"]: row for row in data["studies"]}
    assert studies["E31"]["decision"] == "valid_negative"
    assert studies["E47"]["decision"] == "valid_negative"
    assert studies["E49"]["decision"] == "valid_negative"
    assert studies["E48"]["policy_macro_roc_auc"] > studies["E31"]["primary_auc"]

    raw = RECEIPT.read_text().lower()
    forbidden = (
        "s3://",
        "/home/sagemaker-user",
        "ensemble_weights",
        "teacher_score_array",
        "row_level_prediction",
        "private_prompt",
    )
    assert not any(token in raw for token in forbidden)


def test_employer_docs_link_reproducibility_and_frontier_receipt() -> None:
    readme = (ROOT / "README.md").read_text()
    start = (ROOT / "START_HERE.md").read_text()
    frontier = (ROOT / "docs/POST_CLOSEOUT_FRONTIER.md").read_text()

    assert "docs/REPRODUCIBILITY.md" in readme
    assert "REPRODUCIBILITY.md" in start
    assert "cross_model_frontier_20261006.json" in frontier
