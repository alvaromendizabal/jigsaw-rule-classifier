import json
from pathlib import Path


def test_qwen14b_frontier_checkpoint_is_internally_consistent():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "reports/checkpoints/qwen14b_frontier.json").read_text())
    metrics = data["development_result"]["metrics"]

    assert data["status"] == "completed_not_promoted_as_standalone"
    assert data["execution"]["competition_platform_operations"] == 0
    assert data["development_result"]["decision"]["private_score_used_for_selection"] is False
    assert metrics["qwen14b"]["rule_macro_auc"] < metrics["qwen4b"]["rule_macro_auc"]
    assert metrics["blend_50_50"]["rule_macro_auc"] > metrics["qwen4b"]["rule_macro_auc"]

    expected_gain = (
        metrics["blend_50_50"]["rule_macro_auc"]
        - metrics["qwen4b"]["rule_macro_auc"]
    )
    assert data["development_result"]["blend_gain_vs_qwen4b"] == expected_gain

    for rule, baseline in metrics["qwen4b"]["per_rule_auc"].items():
        assert metrics["blend_50_50"]["per_rule_auc"][rule] > baseline


def test_qwen14b_frontier_public_config_matches_checkpoint():
    root = Path(__file__).resolve().parents[1]
    checkpoint = json.loads((root / "reports/checkpoints/qwen14b_frontier.json").read_text())
    config = json.loads((root / "configs/qwen3_14b_frontier.json").read_text())

    assert config["model_id"] == checkpoint["model"]["model_id"]
    assert config["model_revision"] == checkpoint["model"]["model_revision"]
    assert config["training"]["rank"] == checkpoint["model"]["rank"]
    assert config["training"]["alpha"] == checkpoint["model"]["alpha"]
    assert config["evaluation"]["private_leaderboard_score_used_for_selection"] is False
