import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def test_latest_checkpoint_summary_contract():
    summary = json.loads((ROOT / "reports/majority_submission/summary.json").read_text())
    assert summary["verified_scored_system"]["private_auc"] == 0.91425
    assert summary["candidate"]["status"] == "complete"
    assert summary["candidate"]["public_auc"] == 0.9172
    assert summary["candidate"]["private_auc"] == 0.91288
    assert summary["candidate"]["private_delta_vs_retained"] < 0
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
    svg = sum("image/svg+xml" in out.get("data", {}) for cell in code for out in cell.outputs)
    assert plotly >= 2
    assert svg >= 1
    assert "PLOTLY_SENTINEL_COMPLETE" in text
    assert "LATEST_CHECKPOINT_COMPLETE" in text


def test_closeout_metrics_match_scored_receipt():
    closeout = json.loads((ROOT / "reports/portfolio/closeout.json").read_text())
    receipt = json.loads((ROOT / "reports/checkpoints/kaggle_adaptation.json").read_text())
    final = closeout["final_model"]
    assert closeout["status"] == "complete"
    assert final["private_auc"] == receipt["submission"]["private_score"]
    assert final["public_auc"] == receipt["submission"]["public_score"]
    assert abs(final["private_auc"] - final["baseline_private_auc"] - 0.29469) < 1e-12
    assert abs(final["historical_winner_private_auc"] - final["private_auc"] - 0.01505) < 1e-12
    assert final["late_submission"] is True


def test_closeout_rejects_scored_majority_candidate():
    closeout = json.loads((ROOT / "reports/portfolio/closeout.json").read_text())
    candidate = closeout["supplementary_candidate"]
    assert candidate["submission_ref"] == "56444879"
    assert candidate["last_verified_status"] == "complete"
    assert candidate["public_auc"] == 0.9172
    assert candidate["private_auc"] == 0.91288
    assert candidate["private_delta_vs_retained"] < 0
    assert candidate["promoted"] is False
    assert candidate["blocks_portfolio_closeout"] is False
    assert all(value is False for value in closeout["publication_boundary"].values())


def test_closeout_records_frontier_extension_without_hidden_score_selection():
    closeout = json.loads((ROOT / "reports/portfolio/closeout.json").read_text())
    frontier = closeout["frontier_extension"]
    assert frontier["qwen14b_status"] == "completed_not_promoted_as_standalone"
    assert frontier["qwen14b_policy_macro_auc"] < frontier["qwen4b_reference_policy_macro_auc"]
    assert (
        frontier["fixed_4b14b_blend_policy_macro_auc"]
        > frontier["qwen4b_reference_policy_macro_auc"]
    )
    assert frontier["private_leaderboard_score_used_for_selection"] is False


def test_closeout_source_hashes_are_exact():
    import hashlib

    closeout = json.loads((ROOT / "reports/portfolio/closeout.json").read_text())
    for relative, expected in closeout["sources_sha256"].items():
        assert relative.startswith(("reports/", "configs/"))
        assert ".." not in Path(relative).parts
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected


def test_review_code_uses_only_public_report_libraries():
    import ast

    notebook = nbformat.read(ROOT / "notebooks/27_latest_system_checkpoint.ipynb", as_version=4)
    allowed = {"pathlib", "hashlib", "html", "json", "plotly", "IPython"}
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for node in ast.walk(ast.parse(cell.source)):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] in allowed for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                assert node.module.split(".")[0] in allowed


def test_report_saved_execution_and_chart_scales():
    notebook = nbformat.read(ROOT / "notebooks/27_latest_system_checkpoint.ipynb", as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert [cell.execution_count for cell in code] == list(range(1, len(code) + 1))
    outputs = [output for cell in code for output in cell.outputs]
    text = "".join(output.get("text", "") for output in outputs)
    assert "PLOTLY_SENTINEL_COMPLETE" in text
    assert "LATEST_CHECKPOINT_COMPLETE" in text
    plots = [o.data for o in outputs if "application/vnd.plotly.v1+json" in o.get("data", {})]
    assert len(plots) == 2
    for data in plots:
        figure = data["application/vnd.plotly.v1+json"]
        assert "image/svg+xml" in data
        assert 300 <= figure["layout"]["height"] <= 500
        assert figure["data"][0]["type"] == "scatter"
        low, high = figure["layout"]["xaxis"]["range"]
        assert all(low <= value <= high for value in figure["data"][0]["x"])


def test_public_review_clean_kernel_and_reopen(tmp_path):
    from nbclient import NotebookClient

    path = ROOT / "notebooks/27_latest_system_checkpoint.ipynb"
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=90,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    destination = tmp_path / "review.ipynb"
    nbformat.write(notebook, destination)
    reopened = nbformat.read(destination, as_version=4)
    outputs = [o for c in reopened.cells if c.cell_type == "code" for o in c.outputs]
    assert not any(o.output_type == "error" for o in outputs)
    assert sum("application/vnd.plotly.v1+json" in o.get("data", {}) for o in outputs) == 2
    assert sum("image/svg+xml" in o.get("data", {}) for o in outputs) == 2
    assert "LATEST_CHECKPOINT_COMPLETE" in "".join(o.get("text", "") for o in outputs)
