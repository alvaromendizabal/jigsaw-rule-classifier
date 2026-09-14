from pathlib import Path

from scripts.verify_manual_publication import verify


def test_manual_publication_manifest_and_saved_notebooks():
    result = verify(Path(__file__).resolve().parents[1])
    assert result["notebooks"] == 15
    assert result["plotly"] >= 15
