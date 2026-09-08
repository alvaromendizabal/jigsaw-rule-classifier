"""Leakage boundaries, feature accounting, cached geometry and public-lineage tests."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from jigsaw_rules import research
from jigsaw_rules.data import EXAMPLES
from jigsaw_rules.embeddings import content_key
from jigsaw_rules.representations import (
    TEXT_STAT_NAMES,
    TrainingScreen,
    lexical_candidates,
    semantic_candidates,
    structural_candidates,
    text_statistics,
)
from jigsaw_rules.runtime import atomic_json, digest, stage


def frame(rows=30):
    result = pd.DataFrame(
        {
            "row_id": np.arange(rows),
            "body": [
                f"Can you help with this example number {i}? " + "word " * (i % 4)
                for i in range(rows)
            ],
            "rule": ["Do not offer advice."] * rows,
            "subreddit": [f"community_{i % 3}" for i in range(rows)],
            "rule_violation": np.arange(rows) % 2,
        }
    )
    for i, name in enumerate(EXAMPLES):
        result[name] = [
            f"Provided support {i}: " + "context " * (j % 3 + i + 1) for j in range(rows)
        ]
    return result


def vectors(rows=30, dimensions=8):
    result = np.random.default_rng(11).normal(size=(rows, 5, dimensions))
    return result / np.linalg.norm(result, axis=2, keepdims=True)


@pytest.mark.parametrize("maximum", [0, -1, 1.2, True])
def test_invalid_screen_budgets(maximum):
    with pytest.raises(ValueError, match="positive integer"):
        TrainingScreen(maximum)


@pytest.mark.parametrize("threshold", [0, -0.5, 1.1])
def test_invalid_correlation_threshold(threshold):
    with pytest.raises(ValueError, match="Correlation"):
        TrainingScreen(3, threshold)


def test_screen_accounts_for_every_rejection_and_retained_column():
    rng = np.random.default_rng(23)
    target = np.arange(80) % 2
    random = rng.normal(size=80)
    rare = np.zeros(80)
    rare[0] = 1
    x = np.column_stack([np.ones(80), random, random, target, rare, rng.normal(size=80)])
    names = [f"feature_{i}" for i in range(x.shape[1])]
    screen = TrainingScreen(4).fit(x, target, names)
    reasons = screen.audit_["decisions"]
    assert reasons["constant_or_near_constant"] == 1
    assert reasons["exact_duplicate"] == 1
    assert reasons["perfect_training_separator"] == 1
    assert reasons["rare"] == 1
    assert screen.audit_["retained"] == 2
    assert screen.audit_["rejected"] == 4
    assert sum(reasons.values()) == 6
    assert np.isfinite(screen.catalog_.training_effect_score).all()


def test_sparse_and_dense_screens_agree_when_correlation_checks_are_disabled():
    x = np.random.default_rng(4).uniform(size=(60, 10))
    x[:, 3] = x[:, 2]
    names = [str(i) for i in range(10)]
    target = np.arange(60) % 2
    dense = TrainingScreen(5, None).fit(x, target, names)
    compressed = TrainingScreen(5, None).fit(sparse.csr_matrix(x), target, names)
    np.testing.assert_array_equal(dense.indices_, compressed.indices_)
    assert dense.audit_["decisions"] == compressed.audit_["decisions"]
    np.testing.assert_allclose(
        dense.transform(x, names), compressed.transform(sparse.csr_matrix(x), names).toarray()
    )


def test_validation_transform_does_not_change_screen_or_training_input():
    x = np.random.default_rng(8).normal(size=(80, 12))
    before = x.copy()
    names = [str(i) for i in range(12)]
    screen = TrainingScreen(4).fit(x, np.arange(80) % 2, names)
    decisions = screen.catalog_.copy(deep=True)
    screen.transform(np.full((4, 12), 100000.0), names)
    pd.testing.assert_frame_equal(screen.catalog_, decisions)
    np.testing.assert_array_equal(x, before)
    again = TrainingScreen(4).fit(x, np.arange(80) % 2, names)
    np.testing.assert_array_equal(screen.indices_, again.indices_)


def test_correlation_screen_rejects_training_redundancy():
    rng = np.random.default_rng(20)
    x = rng.normal(size=(90, 4))
    x[:, 1] = x[:, 0] * 2
    screen = TrainingScreen(4).fit(x, np.arange(90) % 2, list("abcd"))
    assert screen.audit_["decisions"]["high_training_correlation"] == 1
    assert screen.audit_["retained"] == 3


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_candidates_fail_before_selection(bad):
    x = np.ones((8, 2))
    x[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        TrainingScreen(2).fit(x, np.arange(8) % 2, ["a", "b"])


def test_screen_requires_binary_labels_and_matching_schema():
    x = np.random.default_rng(2).normal(size=(20, 3))
    with pytest.raises(ValueError, match="binary labels"):
        TrainingScreen(2).fit(x, np.zeros(20), list("abc"))
    with pytest.raises(ValueError, match="unique names"):
        TrainingScreen(2).fit(x, np.arange(20) % 2, ["a", "a", "b"])
    screen = TrainingScreen(2).fit(x, np.arange(20) % 2, list("abc"))
    with pytest.raises(ValueError, match="schema/order"):
        screen.transform(x, list("bac"))
    with pytest.raises(ValueError, match="Fit the training screen"):
        TrainingScreen(2).transform(x, list("abc"))


def test_empty_and_unicode_text_have_finite_explicit_statistics():
    x = text_statistics(pd.Series(["", "WHY??", "no\nlinks https://example.org", "中文 café …"]))
    assert x.shape == (4, len(TEXT_STAT_NAMES))
    assert np.isfinite(x).all()
    assert x[0, TEXT_STAT_NAMES.index("empty_text")] == 1
    assert x[1, TEXT_STAT_NAMES.index("uppercase_fraction")] > 0
    with pytest.raises(ValueError, match="strings"):
        text_statistics(pd.Series([None]))


def test_structural_candidates_are_label_free_and_support_order_invariant():
    data = frame()
    original, names = structural_candidates(data)
    assert original.shape == (30, 1512)
    assert len(set(names)) == len(names)
    swapped = data.copy()
    swapped[list(EXAMPLES)] = data[[EXAMPLES[1], EXAMPLES[0], EXAMPLES[3], EXAMPLES[2]]].to_numpy()
    swapped["rule_violation"] = 1 - swapped.rule_violation
    changed, changed_names = structural_candidates(swapped)
    assert names == changed_names
    np.testing.assert_allclose(original, changed)
    np.testing.assert_allclose(
        original, structural_candidates(data.drop(columns="rule_violation"))[0]
    )


def test_lexical_interactions_keep_a_training_only_vocabulary():
    training, validation = frame(), frame(3)
    validation["body"] = "validationuniquemarker unheardphrase"
    word = TfidfVectorizer().fit(training.body)
    character = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4)).fit(training.body)
    before = dict(word.vocabulary_)
    x, names = lexical_candidates(validation, word, character)
    assert x.shape == (3, 207)
    assert len(set(names)) == 207
    assert np.isfinite(x).all()
    assert word.vocabulary_ == before
    assert "validationuniquemarker" not in before


def test_semantic_geometry_has_six_coordinate_families_and_eight_scalars():
    original = vectors()
    x, names = semantic_candidates(original)
    assert x.shape == (30, 6 * 8 + 32)
    assert len(set(names)) == x.shape[1]
    changed, other_names = semantic_candidates(original[:, [0, 2, 1, 4, 3]])
    assert names == other_names
    np.testing.assert_allclose(x, changed)
    with pytest.raises(ValueError, match="unit L2"):
        semantic_candidates(original * 2)
    with pytest.raises(ValueError, match="aligned"):
        semantic_candidates(original[:, :4])


def cache_shard(cache: Path, name: str, keys: list[str], array: np.ndarray):
    def build(path):
        np.save(path / "vectors.npy", array, allow_pickle=False)
        atomic_json(path / "inputs.json", keys)
        atomic_json(path / "statistics.json", {"truncated_rows": 0})

    return stage(cache, name, build)


def test_embedding_cache_reads_arbitrary_subsets_without_reencoding(tmp_path):
    contract = {"pinned": True}
    atomic_json(tmp_path / "contract.json", contract)
    array = vectors(rows=1)[0, :3]
    cache_shard(tmp_path, "batch_a", ["a", "b", "c"], array)
    (tmp_path / "batch_unfinished.lock").touch()
    result, stats = research.read_embedding_cache(tmp_path, contract, ["c", "a", "c"])
    np.testing.assert_array_equal(result, array[[2, 0, 2]])
    assert stats["newly_encoded_inputs"] == 0
    assert stats["unique_requested_inputs"] == 2
    assert stats["shards_verified"] == 1
    with pytest.raises(ValueError, match="lacks 1"):
        research.read_embedding_cache(tmp_path, contract, ["missing"])
    with pytest.raises(ValueError, match="contract"):
        research.read_embedding_cache(tmp_path, {"pinned": False}, ["a"])


def test_corrupt_embedding_shard_cannot_be_reused(tmp_path):
    atomic_json(tmp_path / "contract.json", {})
    shard = cache_shard(tmp_path, "batch_a", ["a"], vectors(rows=1)[0, :1])
    (shard / "inputs.json").write_text('["different"]')
    with pytest.raises(ValueError, match="checksum"):
        research.read_embedding_cache(tmp_path, {}, ["a"])


def test_conflicting_duplicate_input_hashes_are_rejected(tmp_path):
    atomic_json(tmp_path / "contract.json", {})
    array = vectors(rows=1)[0]
    cache_shard(tmp_path, "batch_a", ["a"], array[:1])
    cache_shard(tmp_path, "batch_b", ["a"], array[1:2])
    with pytest.raises(ValueError, match="Conflicting"):
        research.read_embedding_cache(tmp_path, {}, ["a"])


def test_cache_rejects_inconsistent_vector_dimensions(tmp_path):
    atomic_json(tmp_path / "contract.json", {})
    cache_shard(tmp_path, "batch_a", ["a"], vectors(rows=1, dimensions=4)[0, :1])
    cache_shard(tmp_path, "batch_b", ["b"], vectors(rows=1, dimensions=8)[0, :1])
    with pytest.raises(ValueError, match="dimensionality"):
        research.read_embedding_cache(tmp_path, {}, ["a", "b"])


def test_fold_bank_does_not_learn_from_validation_labels_or_vocabulary(tmp_path):
    training, validation = frame(30), frame(10)
    validation["row_id"] += 1000
    validation["body"] = "validationuniquemarker isolatedtoken"
    rng = np.random.default_rng(13)
    structural = (rng.normal(size=(30, 4)), rng.normal(size=(10, 4)), list("abcd"))
    semantic = (rng.normal(size=(30, 8)), rng.normal(size=(10, 8)), [str(i) for i in range(8)])
    banks = research.fold_bank(
        tmp_path, "first", training, validation, structural, semantic, lambda: None
    )
    validation["rule_violation"] = 1 - validation.rule_violation
    other = research.fold_bank(
        tmp_path, "second", training, validation, structural, semantic, lambda: None
    )
    word, _ = joblib.load(tmp_path / "first_vocabulary/vocabulary.joblib")
    assert "validationuniquemarker" not in word.vocabulary_
    for family in research.BUDGETS:
        first_details = json.loads((banks[family] / "details.json").read_text())
        second_details = json.loads((other[family] / "details.json").read_text())
        assert first_details["catalog_sha256"] == second_details["catalog_sha256"]
        assert (banks[family] / "selected.json").read_bytes() == (
            other[family] / "selected.json"
        ).read_bytes()
        assert first_details["candidates"] == first_details["retained"] + first_details["rejected"]


def test_completed_feature_group_reuse_does_not_refit(tmp_path, monkeypatch):
    training, validation = frame(), frame(10)
    rng = np.random.default_rng(13)
    raw = (rng.normal(size=(30, 3)), rng.normal(size=(10, 3)), list("abc"))
    first = research.fold_bank(tmp_path, "fold", training, validation, raw, raw, lambda: None)
    hashes = {name: digest(path / "catalog.csv") for name, path in first.items()}

    def fail(*args, **kwargs):
        raise AssertionError("Completed feature generation was rerun")

    monkeypatch.setattr(research, "_family", fail)
    monkeypatch.setattr(research, "_vocabulary", fail)
    second = research.fold_bank(tmp_path, "fold", training, validation, raw, raw, lambda: None)
    assert hashes == {name: digest(path / "catalog.csv") for name, path in second.items()}


def test_research_requires_no_private_data_when_evidence_is_absent(tmp_path):
    assert research.research_evidence(tmp_path) is None


def test_public_research_rejects_a_changed_implementation(tmp_path, monkeypatch):
    import jigsaw_rules.review

    monkeypatch.setattr(
        jigsaw_rules.review,
        "public_evidence",
        lambda *args: {"run_id": "reference", "training_sha256": "data"},
    )
    monkeypatch.setattr(research, "implementation_hash", lambda root: "current")
    atomic_json(
        tmp_path / "reports/research/metadata.json",
        {
            "schema": 1,
            "data_kind": "competition",
            "baseline_run": "reference",
            "training_sha256": "data",
            "implementation_sha256": "stale",
            "files": {name: "hash" for name in research.PUBLIC_FILES},
        },
    )
    with pytest.raises(ValueError, match="Stale"):
        research.research_evidence(tmp_path)


def test_implementation_hash_changes_when_representation_source_changes(tmp_path):
    names = (
        "research.py",
        "representations.py",
        "context.py",
        "data.py",
        "embeddings.py",
        "semantic.py",
        "metrics.py",
        "runtime.py",
        "uncertainty.py",
    )
    folder = tmp_path / "src/jigsaw_rules"
    folder.mkdir(parents=True)
    for name in names:
        (folder / name).write_text("# initial\n")
    first = research.implementation_hash(tmp_path)
    (folder / "representations.py").write_text("# changed feature generation\n")
    assert research.implementation_hash(tmp_path) != first
    assert content_key({"a": 1}) == content_key({"a": 1})
