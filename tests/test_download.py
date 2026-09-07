from pathlib import Path
from types import SimpleNamespace

import pytest

from jigsaw_rules.data import FILES, synthetic
from jigsaw_rules.download import download


def test_download_resumes_at_file_boundaries(tmp_path, monkeypatch):
    source = tmp_path / "source"
    synthetic(source)
    target = tmp_path / "target"
    calls = []
    fail = {"once": True}

    def run(command, **kwargs):
        name = command[command.index("-f") + 1]
        calls.append(name)
        if name == "test.csv" and fail["once"]:
            fail["once"] = False
            raise OSError("interrupted download")
        dest = Path(command[command.index("-p") + 1]) / name
        dest.write_bytes((source / name).read_bytes())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("jigsaw_rules.download.shutil.which", lambda name: "kaggle")
    monkeypatch.setattr("jigsaw_rules.download.subprocess.run", run)
    with pytest.raises(OSError):
        download(target)
    download(target)
    assert calls.count("train.csv") == 1
    assert calls.count("test.csv") == 2
    assert all((target / name).exists() for name in FILES)


def test_download_adopts_valid_manual_csvs_without_network(tmp_path, monkeypatch):
    synthetic(tmp_path)
    monkeypatch.setattr("jigsaw_rules.download.shutil.which", lambda name: "kaggle")

    def refuse(*args, **kwargs):
        raise AssertionError("Existing valid files should not be downloaded")

    monkeypatch.setattr("jigsaw_rules.download.subprocess.run", refuse)
    download(tmp_path)
