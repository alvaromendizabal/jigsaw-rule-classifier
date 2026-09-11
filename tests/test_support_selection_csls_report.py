from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scripts import build_support_selection_csls_report as report

ROOT = Path(__file__).resolve().parents[1]


def isolated(tmp_path: Path) -> Path:
    (tmp_path / "reports").mkdir()
    shutil.copytree(
        ROOT / "reports/support_selection_csls",
        tmp_path / "reports/support_selection_csls",
    )
    return tmp_path


def test_csls_public_evidence_and_rows(tmp_path):
    records = report.evidence(isolated(tmp_path))
    rows = report.rows(records)
    assert [r["Changed pairs"] for r in rows] == [114, 268]
    assert rows[1]["CSLS negative max reuse (%)"] > rows[1]["Raw negative max reuse (%)"]
    assert all(r["Gate passed"] is False for r in rows)


def test_csls_evidence_rejects_corruption(tmp_path):
    root = isolated(tmp_path)
    path = root / "reports/support_selection_csls/audit.json"
    value = json.loads(path.read_text())
    value["folds"][0]["changed_pairs"] += 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="checksum"):
        report.evidence(root)


def test_csls_notebook_cells_are_public_aggregate_only():
    cells = report.notebook_cells()
    assert len(cells) == 3
    code = cells[1][1]
    assert "build_support_selection_csls_report" in code
    assert not any(token in code for token in ("np.load", "boto3", "train", "query_targets"))
