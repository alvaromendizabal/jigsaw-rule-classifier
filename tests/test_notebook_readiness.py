"""Authored software fixtures, not competition results."""

import json
from pathlib import Path

import nbformat
import pandas as pd
import pytest

from scripts import notebook_readiness as m


def frames():
    rows = []
    for j, rule in enumerate(("No advertising", "No legal advice")):
        for i in range(8):
            rows.append(
                {
                    "row_id": j * 100 + i,
                    "body": f"private fixture {j} comment {i}",
                    "rule": rule,
                    "subreddit": f"private_community_{j}",
                    "positive_example_1": f"fixture support {j} positive one",
                    "positive_example_2": f"fixture support {j} positive two",
                    "negative_example_1": f"fixture support {j} negative one",
                    "negative_example_2": f"fixture support {j} negative two",
                    "rule_violation": i % 2,
                }
            )
    train = pd.DataFrame(rows)
    test = train.iloc[:2].drop(columns="rule_violation").copy()
    test["row_id"] = [1000, 1001]
    sample = pd.DataFrame({"row_id": test.row_id, "rule_violation": [0.5, 0.5]})
    return train, test, sample


def test_aggregate_only_and_no_model_metrics():
    train, test, sample = frames()
    result = m.profile_frames(train, test, sample)
    encoded = json.dumps(result)
    assert "private fixture" not in encoded
    assert "private_community_" not in encoded
    assert result["model_fits"] == 0
    assert result["new_model_metric"] is None
    assert result["file_rows"]["train.csv"] == 16
    assert result["feature_selection_performed"] is False


@pytest.mark.parametrize("field", ["body", *m.EXAMPLES])
def test_purge_from_each_training_context_field(field):
    train, test, sample = frames()
    train.loc[8, field] = train.loc[0, "body"]
    result = m.profile_frames(train, test, sample)
    advertising = result["fold_preflight"][0]
    assert advertising["training_purged"] == 1
    assert advertising["training_retained"] == 7


def test_support_known_query_excluded():
    train, test, sample = frames()
    train.loc[1, "positive_example_1"] = train.loc[0, "body"]
    result = m.profile_frames(train, test, sample)
    assert result["fold_preflight"][0]["support_known_queries"] == 1
    assert result["fold_preflight"][0]["novel_queries"] == 7


def test_conflicting_support_pair_counted():
    train, test, sample = frames()
    train.loc[0, "negative_example_1"] = train.loc[0, "positive_example_1"]
    result = m.profile_frames(train, test, sample)
    assert result["support_conflicts"][0]["conflicting_text_pairs"] == 1


def test_unicode_duplicates_counted():
    train, test, sample = frames()
    train.loc[1, "body"] = "  " + train.loc[0, "body"].upper() + "  "
    result = m.profile_frames(train, test, sample)
    assert result["policies"][0]["repeated_body_occurrences"] == 1


def test_test_labels_rejected():
    train, test, sample = frames()
    test["rule_violation"] = 0
    with pytest.raises(ValueError, match="target labels"):
        m.profile_frames(train, test, sample)


def test_sample_order_rejected():
    train, test, sample = frames()
    with pytest.raises(ValueError, match="row order"):
        m.profile_frames(train, test, sample.iloc[::-1])


def test_bad_hash_rejected_before_parsing(tmp_path):
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    (raw / "train.csv").write_text("not the original")
    with pytest.raises(ValueError, match="hash mismatch"):
        m.raw_identity(tmp_path)


@pytest.mark.parametrize("name", m.CHARTS)
def test_charts_only_aggregate_counts(name):
    data = m.profile_frames(*frames())
    plot = m.figure(data, name)
    assert plot.data
    assert "private fixture" not in plot.to_json()
    assert "private_community_" not in plot.to_json()


def test_unknown_chart_rejected():
    with pytest.raises(ValueError, match="Unknown chart"):
        m.figure({}, "unregistered")


def test_export_is_self_contained_without_raw_comments(tmp_path):
    summary = m.profile_frames(*frames())
    result = m.save_profile(tmp_path, summary, [m.figure(summary, name) for name in m.CHARTS])
    text = (tmp_path / "reports/data_readiness/dashboard.html").read_text()
    assert "<script src=" not in text
    assert "private fixture" not in text
    assert result["plots"] == 6
    assert result["model_fits"] == 0
    for name, sha in result["files"].items():
        assert m.digest(tmp_path / name) == sha


def test_six_charts_required(tmp_path):
    with pytest.raises(ValueError, match="six charts"):
        m.save_profile(tmp_path, {}, [])


def test_source_hash_ignores_outputs():
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print(1)")])
    before = m.notebook_source(nb)
    nb.cells[0].execution_count = 1
    nb.cells[0].outputs = [nbformat.v4.new_output("stream", text="1\n", name="stdout")]
    assert m.notebook_source(nb) == before
    nb.cells[0].source = "print(2)"
    assert m.notebook_source(nb) != before


def test_unexecuted_notebook_rejected():
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print(1)")])
    with pytest.raises(ValueError, match="unexecuted"):
        m.validate_executed(nb)


def test_corrupt_checkpoint_rejected(tmp_path):
    location = tmp_path / "runs/notebook_readiness"
    location.mkdir(parents=True)
    artifact = tmp_path / "artifact.json"
    artifact.write_text("correct")
    (location / "complete.json").write_text(
        json.dumps({"contract": {}, "files": {"artifact.json": m.digest(artifact)}})
    )
    artifact.write_text("wrong")
    with pytest.raises(ValueError, match="corrupt"):
        m.verify_checkpoint(tmp_path, {})


def test_changed_checkpoint_contract_rejected(tmp_path):
    location = tmp_path / "runs/notebook_readiness"
    location.mkdir(parents=True)
    (location / "complete.json").write_text(json.dumps({"contract": {"old": 1}}))
    with pytest.raises(ValueError, match="contract differs"):
        m.verify_checkpoint(tmp_path, {})


def test_missing_checkpoint_returns_none(tmp_path):
    assert m.verify_checkpoint(tmp_path, {}) is None


def test_code_has_no_training_or_cloud_calls():
    source = Path(m.__file__).read_text()
    assert ".fit(" not in source
    assert "boto3.client(" not in source
    assert "from_pretrained(" not in source
    assert 'transport_encryption="required"' in source
