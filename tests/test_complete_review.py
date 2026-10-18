"""Original final-result receipts stay independently inspectable."""

import json
import shutil
from pathlib import Path

import pytest

from scripts.review_portfolio import review

ROOT = Path(__file__).resolve().parents[1]


def test_final_arithmetic_and_original_evidence():
    result = review(ROOT)
    assert result["status"] == "passed"
    assert result["matched_development"]["gain"] == pytest.approx(0.1052933125268161)
    assert len(result["source_sha256"]) == 11


@pytest.mark.parametrize(
    "section,name", [("support_adaptation", "results.json"), ("baseline", "results.json")]
)
def test_modified_scientific_evidence_is_rejected(tmp_path, section, name):
    shutil.copytree(ROOT / "reports", tmp_path / "reports")
    (tmp_path / "reports" / section / name).write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        review(tmp_path)


def test_changed_submitted_score_is_not_accepted(tmp_path):
    shutil.copytree(ROOT / "reports", tmp_path / "reports")
    path = tmp_path / "reports/checkpoints/kaggle_adaptation.json"
    value = json.loads(path.read_text())
    value["submission"]["private_score"] = 0.99
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Private result changed"):
        review(tmp_path)


def test_metadata_cannot_escape_the_evidence_directory(tmp_path):
    shutil.copytree(ROOT / "reports", tmp_path / "reports")
    path = tmp_path / "reports/baseline/metadata.json"
    value = json.loads(path.read_text())
    value["files"]["../../outside"] = "0" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="escapes"):
        review(tmp_path)
