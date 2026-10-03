import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def checkpoint():
    path = ROOT / "reports/checkpoints/pseudo_supervision_frontier.json"
    return json.loads(path.read_text())


def test_pseudo_supervision_frontier_records_valid_negative_results():
    data = checkpoint()

    assert data["status"] == "completed_valid_negative_frontier"
    assert data["retained_kaggle"]["public_auc"] == 0.91808
    assert data["retained_kaggle"]["private_auc"] == 0.91425
    assert data["retained_kaggle"]["unchanged_by_frontier_research"] is True

    e33 = data["e33_owned_pseudo_supervision"]
    assert e33["status"] == "valid_negative"
    assert e33["completed_units"] == e33["total_units"] == 6
    assert e33["selected_promoted"] is False
    selected_e33 = next(row for row in e33["variants"] if row["id"] == e33["selected"])
    assert selected_e33["delta_vs_incumbent"] < e33["promotion_gate"]["macro_delta_min"]

    e36 = data["e36_external_softlabel_deberta"]
    assert e36["status"] == "valid_negative"
    assert e36["completed_units"] == e36["total_units"] == 6
    assert e36["selected_promoted"] is False
    selected_e36 = next(row for row in e36["variants"] if row["id"] == e36["selected"])
    assert selected_e36["gain_vs_deberta_control"] > 0
    assert selected_e36["delta_vs_incumbent"] < 0
    assert selected_e36["legal_advice_delta"] < 0


def test_pseudo_supervision_frontier_preserves_public_private_boundary():
    data = checkpoint()
    boundary = data["publication_boundary"]
    serialized = json.dumps(data).lower()

    assert boundary["aws_remains_canonical"] is True
    assert data["development_cohort"]["exact_ensemble_weights_public"] is False
    assert "row-level predictions" in boundary["excludes"]
    assert "exact ensemble weights" in boundary["excludes"]
    assert "teacher-score arrays" in boundary["excludes"]
    assert '"weights":' not in serialized


def test_pseudo_supervision_frontier_notebook_has_saved_visual_evidence():
    path = ROOT / "notebooks/30_pseudo_supervision_frontier_review.ipynb"
    notebook = nbformat.read(path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]

    assert code_cells
    assert all(cell.execution_count is not None for cell in code_cells)
    assert all(
        output.get("output_type") != "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )

    mime_types = {
        mime
        for cell in code_cells
        for output in cell.get("outputs", [])
        for mime in output.get("data", {})
    }
    assert "application/vnd.plotly.v1+json" in mime_types
    assert "image/svg+xml" in mime_types

    stdout = "".join(
        output.get("text", "")
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )
    assert "PLOTLY_AND_SVG_RENDERED" in stdout
    assert "FOLLOWING_CELL_EXECUTED" in stdout
    assert "Official Kaggle score unchanged" in stdout
