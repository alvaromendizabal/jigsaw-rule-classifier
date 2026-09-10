"""Completed shards survive a fresh process; corruption fails closed."""

import io
import json
import time
import urllib.error

import pytest

from scripts.support_context_storage import SignedCheckpointStore

BUCKET = "test-bucket"
PREFIX = "experiments/support-context-20260910/checkpoints/test-run/"


def test_snapshot_commit_restore_and_integrity(tmp_path, monkeypatch):
    remote = {}

    class Response(io.BytesIO):
        status = 200

    def request(value, timeout):
        if isinstance(value, str):
            if "snapshot" not in remote:
                raise urllib.error.HTTPError(value, 404, "missing", {}, None)
            return Response(remote["snapshot"])
        assert value.headers["X-amz-server-side-encryption"] == "AES256"
        remote["snapshot"] = value.data
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", request)
    store = SignedCheckpointStore(
        BUCKET,
        [],
        "https://storage.invalid/put",
        "https://storage.invalid/get",
        time.time() + 900,
        tmp_path / "first",
    )
    assert list(store.paginate(Bucket=BUCKET, Prefix=PREFIX)) == [{"Contents": []}]
    features, marker = tmp_path / "features.npz", tmp_path / "complete.json"
    features.write_bytes(b"completed scores")
    marker.write_text(json.dumps({"rows": 32}))
    for path in (features, marker):
        store.upload_file(
            str(path),
            BUCKET,
            PREFIX + "shards/a/" + path.name,
            ExtraArgs={"ServerSideEncryption": "AES256"},
        )
        if path == features:
            assert not remote  # No completion marker before its payload exists.
    resumed = SignedCheckpointStore(
        BUCKET,
        [],
        "https://storage.invalid/put",
        "https://storage.invalid/get",
        time.time() + 900,
        tmp_path / "resumed",
    )
    contents = list(resumed.paginate(Bucket=BUCKET, Prefix=PREFIX))[0]["Contents"]
    assert len(contents) == 2
    recovered = tmp_path / "recovered.npz"
    resumed.download_file(BUCKET, PREFIX + "shards/a/features.npz", str(recovered))
    assert recovered.read_bytes() == features.read_bytes()
    with pytest.raises(ValueError, match="authorized scope"):
        resumed.download_file(BUCKET, "unapproved/input", str(recovered))
    resumed.deadline = time.time() + 59
    with pytest.raises(TimeoutError, match="expiry approaching"):
        resumed.guard()
    # A valid archive with a changed payload is rejected by its stored checksum.
    import tarfile

    source = io.BytesIO(remote["snapshot"])
    corrupt = io.BytesIO()
    with tarfile.open(fileobj=source) as old, tarfile.open(fileobj=corrupt, mode="w:gz") as new:
        for item in old:
            payload = old.extractfile(item).read()
            if item.name.endswith("features.npz"):
                payload = b"corrupted scores"
            item.size = len(payload)
            new.addfile(item, io.BytesIO(payload))
    remote["snapshot"] = corrupt.getvalue()
    bad = SignedCheckpointStore(
        BUCKET,
        [],
        "https://storage.invalid/put",
        "https://storage.invalid/get",
        time.time() + 900,
        tmp_path / "bad",
    )
    with pytest.raises(ValueError, match="checksum differs"):
        list(bad.paginate(Bucket=BUCKET, Prefix=PREFIX))
