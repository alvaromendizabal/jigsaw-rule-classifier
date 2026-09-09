"""Target self-exclusion, text isolation, schema and serialization contracts."""

from pathlib import Path

import joblib
import numpy as np
import pytest

from jigsaw_rules.data import EXAMPLES
from jigsaw_rules.retrieval import PolicyRetrieval, load_plan
from tests.test_expanded import cohort
from tests.test_research import vectors


def plan():
    return load_plan(Path(__file__).resolve().parents[1])


def test_query_policy_labels_cannot_predict_their_own_training_features():
    data, values = cohort(), vectors(120)
    initial, names = PolicyRetrieval(plan()).fit_transform(data, values)
    assert initial.shape == (120, 309)
    assert len(names) == len(set(names)) == 309
    for policy in data.rule.unique():
        mask = data.rule == policy
        changed = data.copy()
        changed.loc[mask, "rule_violation"] = 1 - changed.loc[mask, "rule_violation"]
        after, _ = PolicyRetrieval(plan()).fit_transform(changed, values)
        np.testing.assert_array_equal(initial[mask], after[mask])


@pytest.mark.parametrize("column", ["body", *EXAMPLES])
def test_training_retrieval_purges_each_reference_text_route(column):
    data, values = cohort(), vectors(120)
    data.loc[40, column] = data.loc[0, "body"]
    encoder = PolicyRetrieval(plan())
    encoder.fit_transform(data, values)
    record = next(r for r in encoder.inner_audit_ if r["query_policy"] == data.loc[0, "rule"])
    assert 40 not in record["reference_row_ids"]
    assert set(record["query_row_ids"]).isdisjoint(record["reference_row_ids"])


@pytest.mark.parametrize("column", ["body", *EXAMPLES])
def test_validation_retrieval_rejects_each_unpurged_reference_text_route(column):
    data, values = cohort(), vectors(120)
    training, query = data.iloc[30:].copy(), data.iloc[:30].drop(columns="rule_violation")
    training.loc[40, column] = query.iloc[0].body
    encoder = PolicyRetrieval(plan()).fit(training, values[30:])
    with pytest.raises(ValueError, match="leaks"):
        encoder.transform(query, values[:30])


def test_transform_rejects_query_target_column():
    data, values = cohort(), vectors(120)
    encoder = PolicyRetrieval(plan()).fit(data.iloc[30:], values[30:])
    with pytest.raises(ValueError, match="query targets"):
        encoder.transform(data.iloc[:30], values[:30])


def test_reference_serialization_and_input_permutation_preserve_features(tmp_path):
    data, values = cohort(), vectors(120)
    encoder = PolicyRetrieval(plan()).fit(data.iloc[30:], values[30:])
    query = data.iloc[:30].drop(columns="rule_violation")
    expected, names = encoder.transform(query, values[:30])
    path = tmp_path / "reference.joblib"
    joblib.dump(encoder, path)
    restored = joblib.load(path)
    actual, schema = restored.transform(query, values[:30])
    assert schema == names
    np.testing.assert_array_equal(actual, expected)
    order = np.random.default_rng(2).permutation(len(data) - 30) + 30
    shuffled = PolicyRetrieval(plan()).fit(data.iloc[order], values[order])
    actual, _ = shuffled.transform(query, values[:30])
    np.testing.assert_array_equal(actual, expected)


def test_nonunit_and_nonfinite_embeddings_are_rejected():
    data, values = cohort(), vectors(120)
    with pytest.raises(ValueError, match="unit norm"):
        PolicyRetrieval(plan()).fit(data, values * 10)
    values[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        PolicyRetrieval(plan()).fit(data, values)
