import numpy as np

from jigsaw_rules.data import EXAMPLES, normalize
from jigsaw_rules.model import LexicalClassifier
from jigsaw_rules.splits import make_splits, purged_split


def test_heldout_rule_and_context_purge(dataset):
    train, _, _ = dataset
    train.loc[0, "positive_example_1"] = train.loc[48, "body"]
    for ti, vi, _ in make_splits(train, "heldout_rule", 3, 2025):
        assert not set(train.iloc[ti].rule) & set(train.iloc[vi].rule)
        forbidden = set(train.iloc[vi].body.map(normalize))
        for column in ["body", *EXAMPLES]:
            assert not set(train.iloc[ti][column].map(normalize)) & forbidden


def test_purge_removes_training_support_leak(dataset):
    train, _, _ = dataset
    train.loc[5, "negative_example_1"] = train.loc[0, "body"].upper()
    ti, _, removed = purged_split(train, np.arange(4, len(train)), np.arange(4))
    assert 5 not in ti
    assert removed == 1


def test_group_duplicates_and_determinism(dataset):
    train, _, _ = dataset
    train.loc[2, "body"] = train.loc[0, "body"]
    a = make_splits(train, "seen_rule", 3, 2025)
    b = make_splits(train, "seen_rule", 3, 2025)
    assert sorted(np.concatenate([vi for _, vi, _ in a]).tolist()) == list(range(len(train)))
    for (ti, vi, _), (_, vj, _) in zip(a, b, strict=True):
        assert np.array_equal(vi, vj)
        assert not set(train.iloc[ti].body.map(normalize)) & set(train.iloc[vi].body.map(normalize))


def test_model_never_fits_validation_vocabulary(dataset):
    train, test, _ = dataset
    model = LexicalClassifier().fit(train)
    test["body"] = "validationuniquetoken"
    p = model.predict(test)
    assert "validationuniquetoken" not in model.vectorizer.vocabulary_
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


def test_model_deterministic(dataset):
    train, test, _ = dataset
    np.testing.assert_array_equal(
        LexicalClassifier().fit(train).predict(test), LexicalClassifier().fit(train).predict(test)
    )
