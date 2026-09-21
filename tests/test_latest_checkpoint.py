import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def test_latest_checkpoint_summary_contract():
    summary = json.loads((ROOT / "reports/majority_submission/summary.json").read_text())
    assert summary["verified_scored_system"]["private_auc"] == 0.91425
    assert summary["candidate"]["status"] == "pending"
    assert summary["candidate"]["automatic_promotion"] is False
    assert summary["supervision_audit"]["resolved_majority_pairs"] == 10
    assert summary["supervision_audit"]["discarded_tie_pairs"] == 1
    assert summary["supervision_audit"]["uncontested_label_parity"] is True
    assert summary["supervision_audit"]["forbidden_text_occurrences"] == 0
    assert all(row["purge_passed"] for row in summary["supervision_audit"]["folds"])


def test_latest_checkpoint_notebook_is_executed_and_portable():
    path = ROOT / "notebooks/27_latest_system_checkpoint.ipynb"
    text = path.read_text()
    for forbidden in ("/home/sagemaker-user", "aws_access_key", "positive_example_1"):
        assert forbidden not in text.lower()
    nb = nbformat.read(path, as_version=4)
    code = [cell for cell in nb.cells if cell.cell_type == "code"]
    assert code and all(cell.execution_count is not None for cell in code)
    assert not any(out.output_type == "error" for cell in code for out in cell.outputs)
    plotly = sum(
        "application/vnd.plotly.v1+json" in out.get("data", {})
        for cell in code
        for out in cell.outputs
    )
    svg = sum(
        "image/svg+xml" in out.get("data", {})
        for cell in code
        for out in cell.outputs
    )
    assert plotly >= 2
    assert svg >= 1
    assert "PLOTLY_SENTINEL_COMPLETE" in text
    assert "LATEST_CHECKPOINT_COMPLETE" in text
