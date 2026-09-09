import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jigsaw_rules.data import EXAMPLES, synthetic
from scripts.kaggle_adaptation import adaptation_rows, cached_scores, policy_ranks, prepare_model


def test_canonical_neural_notebook_embeds_exact_sources_without_executing():
    from scripts.build_adapted_notebook import build

    root = Path(__file__).resolve().parents[1]
    notebook = build(root)
    installed = {}

    def install_source(relative, content, expected):
        assert hashlib.sha256(content.encode()).hexdigest() == expected
        installed[relative] = content

    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "notebook", "exec")
        if cell.source.startswith("# Canonical source:"):
            exec(cell.source, {"install_source": install_source})
    for name in ["decision_training.py", "kaggle_adaptation.py", "competition_features.py"]:
        assert installed[f"scripts/{name}"] == (root / "scripts" / name).read_text()
    assert all(cell.get("execution_count") is None for cell in notebook.cells)
    assert build(root) == notebook


def test_only_supplied_new_rule_labels_enter_adaptation(tmp_path):
    synthetic(tmp_path)
    training = pd.read_csv(tmp_path / "train.csv")
    available = pd.read_csv(tmp_path / "test.csv")
    available["rule"] = "An unseen policy"
    for i, column in enumerate(EXAMPLES):
        available[column] = f"Known supplied support {i}"
    pairs, repeats, audit = adaptation_rows(training, available)
    novel = pairs.rule.eq("An unseen policy")
    assert novel.sum() == 4
    assert all(repeat == 2 for repeat, selected in zip(repeats, novel, strict=True) if selected)
    assert set(pairs.loc[novel, "body"]) == {f"Known supplied support {i}" for i in range(4)}
    assert audit["query_body_targets_used"] is False
    with pytest.raises(ValueError, match="target"):
        adaptation_rows(training, available.assign(rule_violation=1))


def test_policy_ranks_preserve_extreme_log_odds_and_row_order():
    margins = np.array([100.0, 101.0, 200.0, -100.0, 200.0])
    rules = np.array(["a", "a", "b", "a", "b"])
    scores = policy_ranks(margins, rules)
    assert scores[1] > scores[0] > scores[3]
    assert scores[2] == scores[4] == 0.5
    order = np.array([4, 1, 0, 3, 2])
    np.testing.assert_equal(policy_ranks(margins[order], rules[order]), scores[order])
    with pytest.raises(ValueError, match="Invalid"):
        policy_ranks([float("nan")], ["a"])


def test_decision_cache_reuses_without_encoder_and_rejects_corruption(tmp_path):
    class Encoder:
        def encode(self, texts):
            return np.array([[1.0, 100.0, 1.0], [1.0, 101.0, 1.0]]), np.zeros((2, 1))

    values = cached_scores(tmp_path, [2, 0], ["one", "two"], Encoder())
    np.testing.assert_equal(cached_scores(tmp_path, [2, 0], ["one", "two"], None), values)
    with pytest.raises(ValueError, match="contract"):
        cached_scores(tmp_path, [0, 2], ["one", "two"], None)
    marker = tmp_path / "complete.json"
    record = json.loads(marker.read_text())
    record["sha256"] = "f" * 64
    marker.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="checksum"):
        cached_scores(tmp_path, [2, 0], ["one", "two"], None)


def test_model_mirror_uses_exact_tokenizer_and_rejects_weight_mismatch(tmp_path):
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    assets = tmp_path / "assets"
    assets.mkdir()
    (mirror / "weights").write_bytes(b"authored software fixture")
    (mirror / "tokenizer_config.json").write_bytes(b"old revision")
    (assets / "tokenizer_config.json").write_bytes(b"canonical revision")
    spec = {
        "files": {
            "weights": {
                "algorithm": "sha256",
                "digest": hashlib.sha256(b"authored software fixture").hexdigest(),
            },
            "tokenizer_config.json": {
                "algorithm": "sha256",
                "digest": hashlib.sha256(b"canonical revision").hexdigest(),
            },
        }
    }
    path = prepare_model(mirror, tmp_path / "prepared", spec, assets)
    assert (path / "tokenizer_config.json").read_bytes() == b"canonical revision"
    (mirror / "weights").write_bytes(b"altered fixture")
    with pytest.raises(ValueError, match="checksum"):
        prepare_model(mirror, path, spec, assets)
