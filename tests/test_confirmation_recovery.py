"""Cloud recovery must preserve completed work and reject corruption before writing."""

import hashlib
import io
import json
import tarfile

import pytest

from jigsaw_rules.runtime import digest
from scripts.recover_confirmation import recover

RUN = "a" * 20
PREFIX = f"runs/confirmation/{RUN}/predictions/"


def bundle(tmp_path, failure=None):
    files = {
        PREFIX + name: b"synthetic" for name in ["predictions.csv", "provenance.json", "audit.json"]
    }
    files[PREFIX + "complete.json"] = json.dumps(
        {
            "files": {
                name.removeprefix(PREFIX): hashlib.sha256(p).hexdigest()
                for name, p in files.items()
            }
        }
    ).encode()
    files["runs/embeddings/new/vectors.npy"] = b"preserved only in downloaded archive"
    if failure == "corrupt":
        files[PREFIX + "predictions.csv"] = b"changed"
    elif failure == "traversal":
        files["logs/../outside"] = b"unsafe"
    elif failure == "target":
        files["runs/confirmation/solution.csv"] = b"target"
    elif failure == "missing":
        del files[PREFIX + "audit.json"]
    archive = tmp_path / "archive.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name, payload in files.items():
            item = tarfile.TarInfo(name)
            item.size = len(payload)
            stream.addfile(item, io.BytesIO(payload))
    return archive


def test_restore_preserves_seed_cache_and_reuses_identical_predictions(tmp_path):
    archive = bundle(tmp_path)
    root = tmp_path / "project"
    cache = root / "runs/embeddings/original/vectors.npy"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"original float32 vectors")
    arguments = (root, archive, RUN, digest(archive), archive.stat().st_size)
    first = recover(*arguments)
    assert recover(*arguments) == first
    assert cache.read_bytes() == b"original float32 vectors"
    assert not (root / "runs/embeddings/new").exists()
    assert first["verified_completed_stages"] == 1
    assert not first["reserved_targets_accessed"]
    assert not first["embedding_caches_modified"]


@pytest.mark.parametrize("failure", ["corrupt", "traversal", "target", "missing", "digest"])
def test_invalid_archive_is_rejected_before_any_restore(tmp_path, failure):
    archive = bundle(tmp_path, failure)
    root = tmp_path / "project"
    expected = "0" * 64 if failure == "digest" else digest(archive)
    with pytest.raises(ValueError):
        recover(root, archive, RUN, expected, archive.stat().st_size)
    assert not root.exists()


def test_existing_different_predictions_are_never_overwritten(tmp_path):
    archive = bundle(tmp_path)
    root = tmp_path / "project"
    path = root / PREFIX / "predictions.csv"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"previously frozen predictions")
    with pytest.raises(ValueError, match="different existing"):
        recover(root, archive, RUN, digest(archive), archive.stat().st_size)
    assert path.read_bytes() == b"previously frozen predictions"
    assert not (path.parent / "complete.json").exists()
