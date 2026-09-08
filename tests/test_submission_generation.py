"""User-owned generation, bounded downloads, and failure recovery."""

import json

import pandas as pd
import pytest

from jigsaw_rules.data import validate_submission
from jigsaw_rules.model import LexicalClassifier
from jigsaw_rules.runtime import digest
from jigsaw_rules.submission import download_link, generate_submission


@pytest.fixture
def generate(tmp_path, dataset):
    def execute(**kwargs):
        return generate_submission(
            tmp_path / "data/raw",
            tmp_path / "output",
            tmp_path / "runs/submission_cache",
            source_sha256="a" * 64,
            batch_size=2,
            **kwargs,
        )

    return execute


def test_generation_validates_and_downloads_user_output(generate, dataset):
    path = generate()
    validate_submission(pd.read_csv(path), dataset[2])
    manifest = json.loads((path.parent / "submission_manifest.json").read_text())
    assert manifest["submission_sha256"] == digest(path)
    assert manifest["synthetic"] is True
    link = download_link(path)
    assert 'download="submission.csv"' in link
    assert 'href="data:text/csv;base64,' in link
    assert "https://" not in link


def test_completed_generation_neither_refits_nor_predicts(generate, monkeypatch):
    original = generate().read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed inference must be reused")

    monkeypatch.setattr(LexicalClassifier, "fit", forbidden)
    monkeypatch.setattr(LexicalClassifier, "predict", forbidden)
    assert generate().read_bytes() == original


def test_interrupted_prediction_retains_completed_model_and_batches(
    generate, tmp_path, monkeypatch
):
    predict = LexicalClassifier.predict
    calls = []

    def interrupt(self, frame):
        calls.append(len(frame))
        if len(calls) == 2:
            raise RuntimeError("Authored interruption")
        return predict(self, frame)

    monkeypatch.setattr(LexicalClassifier, "predict", interrupt)
    with pytest.raises(RuntimeError, match="Authored interruption"):
        generate()
    assert not (tmp_path / "output/submission.csv").exists()
    assert len(list((tmp_path / "runs/submission_cache").glob("*/model/complete.json"))) == 1
    assert len(list((tmp_path / "runs/submission_cache").glob("*/batch_*/complete.json"))) == 1

    def no_fit(*args, **kwargs):
        raise AssertionError("The completed model must be retained")

    monkeypatch.setattr(LexicalClassifier, "fit", no_fit)
    monkeypatch.setattr(LexicalClassifier, "predict", predict)
    assert generate().is_file()


def test_corrupt_batch_is_recomputed_without_refitting(generate, tmp_path, monkeypatch):
    original = generate().read_bytes()
    batch = next((tmp_path / "runs/submission_cache").glob("*/batch_*/predictions.csv"))
    batch.write_text("corruption")

    def no_fit(*args, **kwargs):
        raise AssertionError("The intact model must not be refitted")

    monkeypatch.setattr(LexicalClassifier, "fit", no_fit)
    assert generate().read_bytes() == original


def test_changed_input_invalidates_generation_cache(generate, tmp_path, monkeypatch):
    generate()
    with (tmp_path / "data/raw/train.csv").open("a") as stream:
        stream.write("\n")

    def stop_fit(*args, **kwargs):
        raise RuntimeError("New data requires a new model")

    monkeypatch.setattr(LexicalClassifier, "fit", stop_fit)
    with pytest.raises(RuntimeError, match="New data"):
        generate()


def test_download_refuses_tampered_submission(generate):
    path = generate()
    path.write_text("row_id,rule_violation\n1,0.99\n")
    with pytest.raises(ValueError, match="checksum"):
        download_link(path)


def test_large_download_uses_bounded_fallback(generate):
    link = download_link(generate(), max_bytes=1)
    assert "file browser" in link
    assert "base64" not in link


@pytest.mark.parametrize("size,source", [(0, "a" * 64), (1, "bad")])
def test_invalid_generation_contract_is_rejected(tmp_path, size, source):
    with pytest.raises(ValueError, match="batch_size"):
        generate_submission(tmp_path, tmp_path, tmp_path, source_sha256=source, batch_size=size)
