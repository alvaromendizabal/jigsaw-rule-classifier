"""New-cohort leakage boundaries, cache recovery and end-to-end study contracts."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jigsaw_rules import expanded
from jigsaw_rules.data import EXAMPLES
from jigsaw_rules.embeddings import content_key, encode_cached
from jigsaw_rules.expanded_embeddings import encode_batches, prepare_plan, verified_keys
from jigsaw_rules.research import read_embedding_cache
from jigsaw_rules.runtime import digest
from tests.helpers import TestEncoder
from tests.test_research import frame, vectors


def cohort():
    data = frame(120)
    data["rule"] = [f"Policy {i // 30}: no unsolicited advice" for i in range(len(data))]
    return data


def test_cache_extension_reuses_individual_hashes_when_batches_change(tmp_path):
    encoder = TestEncoder()
    original, _ = encode_cached(tmp_path, ["old-a", "old-b", "old-c"], encoder, shard_size=2)
    contract = encoder.contract
    cache = tmp_path / "runs/embeddings" / content_key(contract)[:20]
    originals = {p: digest(p) for p in cache.glob("batch_*/*")}
    requested = ["old-c", "new-a", "old-a", "new-b", "old-b"]
    plan = prepare_plan(tmp_path, requested, contract, 2)
    assert json.loads((plan / "audit.json").read_text())["already_cached"] == 3
    batches = json.loads((plan / "batches.json").read_text())
    assert sorted(t for b in batches for t in b) == ["new-a", "new-b"]
    encode_batches(tmp_path, plan, batches, encoder, 0)
    assert all(digest(p) == sha for p, sha in originals.items())
    values, _ = read_embedding_cache(cache, contract, [content_key(t) for t in requested])
    np.testing.assert_array_equal(values[[2, 4, 0]], original)
    before = encoder.calls
    encode_batches(tmp_path, plan, batches, encoder, 0)
    assert encoder.calls == before
    assert len(verified_keys(cache, contract)) == 5


def test_embedding_interruption_preserves_completed_shards(tmp_path):
    encoder = TestEncoder(fail_on_call=2)
    plan = prepare_plan(tmp_path, ["a", "b", "c", "d"], encoder.contract, 2)
    batches = json.loads((plan / "batches.json").read_text())
    with pytest.raises(RuntimeError, match="Simulated interruption"):
        encode_batches(tmp_path, plan, batches, encoder, 0)
    retry = TestEncoder()
    result = encode_batches(tmp_path, plan, batches, retry, 0)
    assert result["encoded"] == result["reused"] == 2
    assert retry.calls == 1


def test_corrupt_existing_vector_fails_before_extending_cache(tmp_path):
    encoder = TestEncoder()
    encode_cached(tmp_path, ["original"], encoder)
    path = next((tmp_path / "runs/embeddings").glob("*/batch_*/vectors.npy"))
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        prepare_plan(tmp_path, ["original", "new"], encoder.contract, 2)


def test_conflict_removal_uses_only_training_labels():
    training, validation = frame(30), frame(8)
    training.loc[1, "body"] = training.loc[0, "body"]
    validation["rule_violation"] = "PROTECTED_TARGET_DO_NOT_INTERPRET"
    retained = expanded.filter_training(training, validation, "exclude_conflicts")
    assert set(retained.row_id) == set(range(2, 30))
    validation["rule_violation"] = -100
    pd.testing.assert_frame_equal(
        retained, expanded.filter_training(training, validation, "exclude_conflicts")
    )


@pytest.mark.parametrize("column", ["body", *EXAMPLES])
def test_split_validator_rejects_every_training_text_leak(column):
    data = cohort()
    records = expanded.design(data, {"seen_rule_folds": 2, "seed": 2025})
    a = records["seen_rule"][0]
    data.loc[a["train"][0], column] = data.loc[a["valid"][0], "body"]
    with pytest.raises(ValueError, match="leaks"):
        expanded.validate_splits(data, records)


def test_split_validator_rejects_missing_and_duplicate_rows():
    data = cohort()
    records = expanded.design(data, {"seen_rule_folds": 2, "seed": 2025})
    malformed = copy.deepcopy(records)
    malformed["seen_rule"][0]["valid"].pop()
    with pytest.raises(ValueError, match="each row"):
        expanded.validate_splits(data, malformed)
    records["heldout_rule"][0]["train"].append(records["heldout_rule"][0]["valid"][0])
    with pytest.raises(ValueError, match="overlap"):
        expanded.validate_splits(data, records)


def test_prediction_alignment_rejects_wrong_labels_and_duplicate_ids():
    data = cohort()
    prediction = data[["row_id", "rule", "rule_violation"]].copy()
    prediction["probability"] = np.linspace(0.1, 0.9, len(data))
    expected = prediction.probability.to_numpy()
    np.testing.assert_array_equal(
        expanded.aligned_predictions(data, prediction.sample(frac=1, random_state=2)), expected
    )
    with pytest.raises(ValueError, match="one prediction"):
        expanded.aligned_predictions(data, pd.concat([prediction, prediction.iloc[:1]]))
    prediction.loc[0, "rule_violation"] = 1 - prediction.loc[0, "rule_violation"]
    with pytest.raises(ValueError, match="lineage"):
        expanded.aligned_predictions(data, prediction)


def test_equal_group_weight_is_invariant_to_repeated_identical_annotation():
    data = frame(12)
    p = np.linspace(0.1, 0.9, len(data))
    original = expanded.weighted_metrics(data, p)
    duplicated = pd.concat([data, data.iloc[[0, 0, 0]]], ignore_index=True)
    actual = expanded.weighted_metrics(duplicated, np.r_[p, [p[0]] * 3])
    for metric in ("rule_macro_auc", "log_loss", "brier"):
        assert actual[metric] == pytest.approx(original[metric])


def test_support_stress_is_deterministic_and_does_not_mutate_vectors():
    data, values = cohort(), vectors(120)
    before = values.copy()
    result = expanded.support_stress(data, values, 2025)
    assert result == expanded.support_stress(data, values, 2025)
    np.testing.assert_array_equal(values, before)


def test_complete_small_study_and_resume_without_refitting(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    plan = expanded.load_plan(root)
    plan["seen_rule_folds"], plan["bootstrap_draws"] = 2, 100
    data = cohort()
    directory = tmp_path / "runs/expanded/software-test"
    expanded.execute_study(tmp_path, directory, data, vectors(120), plan, {"software_test": True})
    audit = json.loads((directory / "review/audit.json").read_text())
    assert audit["primary_fitted_models"] == 34 * 6
    assert audit["confirmation_labels_accessed"] is False
    assert all(item["reused_primary"] for item in audit["training_sensitivities"])
    oof = pd.read_csv(directory / "review/oof.csv")
    assert oof.groupby(["protocol", "model"]).size().eq(len(data)).all()
    records = expanded.design(data, plan)
    with pytest.raises(ValueError, match="model coverage"):
        expanded.validate_oof(data, oof[oof.model != "comment_only"], plan, records)
    malformed = oof.copy()
    malformed.loc[0, "fold"] = 999
    with pytest.raises(ValueError, match="saved validation"):
        expanded.validate_oof(data, malformed, plan, records)
    # Rendering uses the actual software-study schema; never publish these scores.
    import shutil

    from scripts import build_expanded_report

    public = tmp_path / "reports/expanded"
    public.mkdir(parents=True)
    (public / "metadata.json").write_text('{"software_test": true}')
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(
        root / "scripts/build_expanded_report.py", tmp_path / "scripts/build_expanded_report.py"
    )
    evidence = {
        name.removesuffix(".json"): json.loads((directory / "review" / name).read_text())
        for name in expanded.PUBLIC_FILES
    }
    monkeypatch.setattr(build_expanded_report, "expanded_evidence", lambda root: evidence)
    build_expanded_report.build(tmp_path)
    assert len(build_expanded_report.verify_figures(tmp_path)["files"]) == 8
    (public / "ablation.svg").write_text("changed")
    with pytest.raises(ValueError, match="figure contract"):
        build_expanded_report.verify_figures(tmp_path)
    from jigsaw_rules import retrieval

    (tmp_path / "configs").mkdir()
    shutil.copyfile(root / "configs/retrieval.json", tmp_path / "configs/retrieval.json")
    evidence["metadata"] = {
        "run_id": directory.name,
        "private_checkpoint_sha256": digest(directory / "review/complete.json"),
    }
    monkeypatch.setattr(retrieval, "expanded_evidence", lambda root: evidence)
    monkeypatch.setattr(retrieval, "load_development", lambda root: data)
    monkeypatch.setattr(retrieval, "development_plan", lambda root: plan)
    monkeypatch.setattr(
        retrieval, "cached_vectors", lambda root, frame: (vectors(120), {"software_test": True})
    )
    retrieval_run = retrieval.run_retrieval(tmp_path)
    retrieval_oof = pd.read_csv(retrieval_run / "review/oof.csv")
    assert len(retrieval_oof) == 120 * 5 * 2
    assert len(list(retrieval_run.glob("*_model_*/complete.json"))) == 30
    assert retrieval.retrieval_evidence(tmp_path) is not None
    before = {p: digest(p) for p in directory.glob("*/complete.json")}

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed model was refitted")

    monkeypatch.setattr(expanded, "_model", forbidden)
    monkeypatch.setattr(expanded, "_reference", forbidden)
    monkeypatch.setattr(expanded, "fit_lexical", forbidden)
    monkeypatch.setattr(retrieval, "_model", forbidden)
    assert retrieval.run_retrieval(tmp_path) == retrieval_run
    expanded.execute_study(tmp_path, directory, data, vectors(120), plan, {"software_test": True})
    assert all(digest(p) == sha for p, sha in before.items())


def test_preregistered_config_rejects_unrecorded_changes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "configs").mkdir()
    text = (root / "configs/expanded.json").read_text().replace('"seed": 2025', '"seed": 2026')
    (tmp_path / "configs/expanded.json").write_text(text)
    with pytest.raises(ValueError, match="pre-score"):
        expanded.load_plan(tmp_path)


def test_cloud_archive_rejects_traversal_before_writing(tmp_path):
    import io
    import tarfile

    from scripts.expanded_processing import extract

    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        info = tarfile.TarInfo("../escaped")
        info.size = 1
        stream.addfile(info, io.BytesIO(b"x"))
    root = tmp_path / "output"
    root.mkdir()
    with pytest.raises(ValueError, match="safe regular"):
        extract(archive, root)
    assert not (tmp_path / "escaped").exists()
