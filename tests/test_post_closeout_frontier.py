import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/checkpoints/post_closeout_frontier_20261004.json"


def test_frontier_receipt_is_aggregate_and_consistent():
    data = json.loads(REPORT.read_text())

    assert data["schema"] == 1
    assert data["evaluation"]["metric"] == "policy_macro_roc_auc"
    assert data["evaluation"]["direction"] == "higher_is_better"
    assert data["evaluation"]["rows"] == 881
    assert data["evaluation"]["policies"] == 2
    assert data["evaluation"]["leaderboard_comparable"] is False

    studies = {row["id"]: row for row in data["studies"]}
    assert studies["llama_support_adaptation"]["status"] == "preserved_unpromoted"
    assert studies["semantic_support_retrieval"]["status"] == "valid_negative"
    assert studies["nli_clause_decomposition"]["status"] == "valid_negative"

    boundary = data["public_boundary"]
    assert boundary["aggregate_metrics"] is True
    assert boundary["aggregate_decisions"] is True
    assert boundary["raw_comments"] is False
    assert boundary["row_level_predictions"] is False
    assert boundary["exact_ensemble_weights"] is False
    assert boundary["model_or_optimizer_state"] is False
    assert boundary["teacher_score_arrays"] is False
