import json
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]


def test_five_model_frontier_checkpoint_is_consistent():
    data = json.loads((ROOT / "reports/checkpoints/five_model_frontier.json").read_text())

    assert data["status"] == "development_candidate_promoted_pending_official_score"
    assert data["retained_kaggle"]["public_auc"] == 0.91808
    assert data["retained_kaggle"]["private_auc"] == 0.91425
    assert data["retained_kaggle"]["unchanged_by_frontier_research"] is True

    full = data["full_oof"]
    assert full["fixed_insertion_macro_auc"] > full["incumbent_policy_transfer_macro_auc"]
    assert full["per_policy_gain"]["advertising"] > 0
    assert full["per_policy_gain"]["legal_advice"] > 0
    assert full["grouped_bootstrap"]["probability_positive"] >= 0.95

    compact = data["compression_study"]
    assert compact["status"] == "valid_negative"
    assert compact["promoted"] is False
    assert compact["per_policy_gain"]["legal_advice"] < -0.003

    global_result = data["global_deployment_validation"]
    expected_gain = (
        global_result["promoted_five_model_global_macro_auc"]
        - global_result["prior_four_model_global_macro_auc"]
    )
    assert abs(global_result["gain"] - expected_gain) < 1e-6
    assert global_result["per_policy_gain"]["advertising"] > 0
    assert global_result["per_policy_gain"]["legal_advice"] > 0
    assert global_result["grouped_bootstrap"]["ci_95"][0] > 0
    assert global_result["grouped_bootstrap"]["probability_positive"] >= 0.95
    assert global_result["private_leaderboard_score_used_for_selection"] is False


def test_five_model_frontier_preserves_public_private_boundary():
    data = json.loads((ROOT / "reports/checkpoints/five_model_frontier.json").read_text())
    serialized = json.dumps(data).lower()

    assert data["models"]["exact_deployment_weights_public"] is False
    assert data["global_deployment_validation"]["exact_weights_public"] is False
    assert "row-level predictions" in data["publication_boundary"]["excludes"]
    assert "exact ensemble weights" in data["publication_boundary"]["excludes"]
    assert '"weights":' not in serialized


def test_five_model_frontier_notebook_has_saved_visual_evidence():
    notebook = nbformat.read(ROOT / "notebooks/29_five_model_frontier_review.ipynb", as_version=4)

    assert all(
        output.get("output_type") != "error"
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
    )
    assert all(
        cell.get("execution_count") is not None
        for cell in notebook.cells
        if cell.cell_type == "code"
    )

    mime_types = {
        mime
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        for mime in output.get("data", {})
    }
    assert "application/vnd.plotly.v1+json" in mime_types
    assert "image/svg+xml" in mime_types

    stdout = "".join(
        "".join(output.get("text", []))
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )
    assert "PLOTLY_AND_SVG_RENDERED" in stdout
    assert "FOLLOWING_CELL_EXECUTED" in stdout
    assert "Official Kaggle score unchanged" in stdout
