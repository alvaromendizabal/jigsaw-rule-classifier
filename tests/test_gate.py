from pathlib import Path

import pytest

from jigsaw_rules.gate import feature_gate, require_feature_completion


def test_real_feature_gate_cannot_confuse_broad_search_with_confirmation():
    root = Path(__file__).resolve().parents[1]
    gate = feature_gate(root)
    completed = (root / "reports/expanded/metadata.json").exists()
    assert gate["labeled_rules"] == (4 if completed else 2)
    assert gate["training_rows"] == (11135 if completed else 2029)
    assert gate["historical_reference"]["labeled_rules"] == 2
    assert gate["status"] == "open"
    assert not gate["final_training_authorized_by_evidence"]
    with pytest.raises(ValueError, match="Feature research gate is open"):
        require_feature_completion(root)
