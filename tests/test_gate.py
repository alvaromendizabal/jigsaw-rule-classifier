from pathlib import Path

import pytest

from jigsaw_rules import gate as gate_module
from jigsaw_rules.gate import feature_gate, require_feature_completion


def test_real_feature_gate_cannot_confuse_broad_search_with_confirmation():
    root = Path(__file__).resolve().parents[1]
    gate = feature_gate(root)
    completed = (root / "reports/expanded/metadata.json").exists()
    assert gate["labeled_rules"] == (4 if completed else 2)
    assert gate["training_rows"] == (11135 if completed else 2029)
    assert gate["historical_reference"]["labeled_rules"] == 2
    closed = (root / "reports/feature_decision/metadata.json").exists()
    assert gate["feature_research_complete"] == closed
    assert gate["final_training_authorized_by_evidence"] == closed
    assert not gate["production_promotion_authorized_by_evidence"]
    assert gate["status"] == ("ready_for_final_model_validation" if closed else "open")
    if closed:
        require_feature_completion(root)
        assert gate["selected_transfer_representation"] == "qwen_centroid"
    else:
        with pytest.raises(ValueError, match="Feature research gate is open"):
            require_feature_completion(root)


def test_missing_study_cannot_be_hidden_by_a_published_stopping_decision(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(gate_module, "formatting_evidence", lambda root: None)
    assert not feature_gate(root)["final_training_authorized_by_evidence"]
    with pytest.raises(ValueError, match="Feature research gate is open"):
        require_feature_completion(root)
