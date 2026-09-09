"""Regression coverage for the competition mount used by the live Kaggle run."""

import pytest

from scripts.kaggle_paths import submission_input_root


@pytest.mark.parametrize("prefix", ["competitions/", ""])
def test_official_mount_layouts(tmp_path, monkeypatch, prefix):
    monkeypatch.delenv("JIGSAW_KAGGLE_INPUT", raising=False)
    kaggle = tmp_path / "input"
    expected = kaggle / prefix / "jigsaw-agile-community-rules"
    expected.mkdir(parents=True)
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        (expected / name).touch()
    assert submission_input_root(tmp_path, kaggle) == expected


def test_local_checkout_and_explicit_override(tmp_path, monkeypatch):
    monkeypatch.delenv("JIGSAW_KAGGLE_INPUT", raising=False)
    absent_kaggle = tmp_path / "absent"
    assert submission_input_root(tmp_path, absent_kaggle) == tmp_path / "data/raw"
    monkeypatch.setenv("JIGSAW_KAGGLE_INPUT", str(tmp_path / "chosen"))
    assert submission_input_root(tmp_path, absent_kaggle) == tmp_path / "chosen"


def test_incomplete_attachment_gives_actionable_error(tmp_path, monkeypatch):
    monkeypatch.delenv("JIGSAW_KAGGLE_INPUT", raising=False)
    incomplete = tmp_path / "competitions/jigsaw-agile-community-rules"
    incomplete.mkdir(parents=True)
    (incomplete / "train.csv").touch()
    with pytest.raises(FileNotFoundError, match="Attach the official Jigsaw"):
        submission_input_root(tmp_path, tmp_path)
