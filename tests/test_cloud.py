import hashlib
import io
import json

import pytest

from jigsaw_rules.cloud import backup, restore


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.uploads = []
        self.fail_objects = False

    def list_objects_v2(self, *, Bucket, Prefix, MaxKeys):
        return {"Contents": [{"Key": k} for k in self.objects if k.startswith(Prefix)][:MaxKeys]}

    def upload_fileobj(self, stream, bucket, key, ExtraArgs):
        if self.fail_objects and key.startswith("objects/"):
            raise OSError("network interrupted")
        self.objects[key] = stream.read()
        self.uploads.append(key)

    def get_object(self, *, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key])}


def test_backup_restore_integrity_and_reuse(tmp_path):
    root = tmp_path / "project"
    data = root / "data/raw/train.csv"
    data.parent.mkdir(parents=True)
    data.write_bytes(b"test bytes")
    (root / ".env").write_text("never upload this")
    cloud = FakeS3()
    backup(root, "bucket", "region", cloud)
    objects = [k for k in cloud.uploads if k.startswith("objects/")]
    backup(root, "bucket", "region", cloud)
    assert [k for k in cloud.uploads if k.startswith("objects/")] == objects
    manifest = json.loads(cloud.objects["latest.json"])
    assert ".env" not in manifest["files"]
    new = tmp_path / "restored"
    assert restore(new, "bucket", "region", cloud) == 1
    assert (new / "data/raw/train.csv").read_bytes() == b"test bytes"


def test_interrupted_upload_keeps_previous_manifest(tmp_path):
    file = tmp_path / "runs/result.json"
    file.parent.mkdir()
    file.write_text("first")
    cloud = FakeS3()
    backup(tmp_path, "bucket", "region", cloud)
    previous = cloud.objects["latest.json"]
    file.write_text("second")
    cloud.fail_objects = True
    with pytest.raises(OSError):
        backup(tmp_path, "bucket", "region", cloud)
    assert cloud.objects["latest.json"] == previous


@pytest.mark.parametrize(
    "path", ["../escape", "/absolute", "runs/../../escape", "configs/local.json"]
)
def test_restore_rejects_unsafe_paths(tmp_path, path):
    cloud = FakeS3()
    cloud.objects["latest.json"] = json.dumps({"schema": 1, "files": {path: "a" * 64}}).encode()
    with pytest.raises(ValueError, match="Unsafe"):
        restore(tmp_path, "bucket", "region", cloud)


def test_restore_rejects_corruption_and_conflicts(tmp_path):
    cloud = FakeS3()
    sha = hashlib.sha256(b"good").hexdigest()
    cloud.objects["latest.json"] = json.dumps({"schema": 1, "files": {"runs/x": sha}}).encode()
    cloud.objects[f"objects/{sha}"] = b"bad"
    with pytest.raises(ValueError, match="checksum"):
        restore(tmp_path, "bucket", "region", cloud)
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs/x").write_bytes(b"local work")
    with pytest.raises(FileExistsError):
        restore(tmp_path, "bucket", "region", cloud)
