import pytest

from jigsaw_rules.data import load_data, synthetic


@pytest.fixture
def dataset(tmp_path):
    synthetic(tmp_path / "data/raw")
    return load_data(tmp_path / "data/raw")
