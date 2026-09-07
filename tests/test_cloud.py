import hashlib
import io
import json

import pytest
from botocore.exceptions import ClientError

from jigsaw_rules.cloud import backup, restore


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.uploads = []
        self.fail_objects = False
        self.before_latest = None
        self.after_upload = None
        self.conditions = []

    def list_objects_v2(self, *, Bucket, Prefix, MaxKeys):
        return {"Contents": [{"Key": k} for k in self.objects if k.startswith(Prefix)][:MaxKeys]}

    def upload_fileobj(self, stream, bucket, key, ExtraArgs):
        if self.fail_objects and key.startswith("objects/"):
            raise OSError("network interrupted")
        self.objects[key] = stream.read()
        self.uploads.append(key)
        if self.after_upload:
            self.after_upload(key)

    def get_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key]), "ETag": self.etag(Key)}

    def etag(self, key):
        return '"' + hashlib.sha256(self.objects[key]).hexdigest() + '"'

    def put_object(self, *, Bucket, Key, Body, ServerSideEncryption, ContentType, **condition):
        self.conditions.append(condition)
        if self.before_latest:
            self.before_latest()
        if (condition.get("IfNoneMatch") == "*" and Key in self.objects) or (
            "IfMatch" in condition
            and (Key not in self.objects or self.etag(Key) != condition["IfMatch"])
        ):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[Key] = Body
        self.uploads.append(Key)
        return {"ETag": self.etag(Key)}


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


def test_partial_workspace_cannot_replace_complete_snapshot(tmp_path):
    source = tmp_path / "complete"
    artifact = source / "runs/completed/result.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("verified result")
    cloud = FakeS3()
    backup(source, "bucket", "region", cloud)
    previous = cloud.objects["latest.json"]
    uploads = cloud.uploads.copy()
    partial = tmp_path / "partial"
    (partial / "runs").mkdir(parents=True)
    (partial / "runs/new.json").write_text("new work")
    with pytest.raises(FileNotFoundError, match="would omit 1"):
        backup(partial, "bucket", "region", cloud)
    assert cloud.objects["latest.json"] == previous
    assert cloud.uploads == uploads
    restore(partial, "bucket", "region", cloud)
    backup(partial, "bucket", "region", cloud)
    assert set(json.loads(cloud.objects["latest.json"])["files"]) == {
        "runs/completed/result.json",
        "runs/new.json",
    }


def test_first_and_later_snapshot_use_conditional_writes(tmp_path):
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs/result.json").write_text("first")
    cloud = FakeS3()
    backup(tmp_path, "bucket", "region", cloud)
    etag = cloud.etag("latest.json")
    backup(tmp_path, "bucket", "region", cloud)
    assert cloud.conditions == [{"IfNoneMatch": "*"}, {"IfMatch": etag}]


@pytest.mark.parametrize("existing", [False, True])
def test_competing_backup_keeps_other_writer_snapshot(tmp_path, existing):
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs/result.json").write_text("work")
    cloud = FakeS3()
    if existing:
        backup(tmp_path, "bucket", "region", cloud)
    competing = b'{"concurrent": "writer"}'
    cloud.before_latest = lambda: cloud.objects.update({"latest.json": competing})
    with pytest.raises(RuntimeError, match="Another backup changed"):
        backup(tmp_path, "bucket", "region", cloud)
    assert cloud.objects["latest.json"] == competing
    assert any(k.startswith("snapshots/") for k in cloud.objects)


@pytest.mark.parametrize("code", ["AccessDenied", "NoSuchBucket", "InternalError"])
def test_remote_read_errors_never_create_new_history(tmp_path, code):
    class UnavailableS3(FakeS3):
        def get_object(self, **kwargs):
            raise ClientError({"Error": {"Code": code}}, "GetObject")

    (tmp_path / "runs").mkdir()
    (tmp_path / "runs/result.json").write_text("work")
    cloud = UnavailableS3()
    with pytest.raises(ClientError):
        backup(tmp_path, "bucket", "region", cloud)
    assert cloud.uploads == []


@pytest.mark.parametrize(
    "manifest",
    [
        {"schema": 2, "files": {"runs/x": "a" * 64}},
        {"schema": 1, "files": {}},
        {"schema": 1, "files": []},
        {"schema": 1, "files": {"runs/../x": "a" * 64}},
        {"schema": 1, "files": {"runs//x": "a" * 64}},
        {"schema": 1, "files": {"runs/x": "not-a-hash"}},
        {"schema": 1, "files": {"runs/x": 12}},
    ],
)
def test_invalid_remote_manifest_is_rejected_before_upload(tmp_path, manifest):
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs/x").write_text("work")
    cloud = FakeS3()
    previous = json.dumps(manifest).encode()
    cloud.objects["latest.json"] = previous
    with pytest.raises(ValueError):
        backup(tmp_path, "bucket", "region", cloud)
    assert cloud.uploads == []
    assert cloud.objects["latest.json"] == previous


def test_missing_etag_prevents_unconditional_write(tmp_path):
    class NoEtagS3(FakeS3):
        def get_object(self, **kwargs):
            response = super().get_object(**kwargs)
            response.pop("ETag")
            return response

    (tmp_path / "runs").mkdir()
    (tmp_path / "runs/x").write_text("work")
    cloud = NoEtagS3()
    cloud.objects["latest.json"] = json.dumps(
        {"schema": 1, "files": {"runs/x": "a" * 64}}
    ).encode()
    with pytest.raises(ValueError, match="ETag"):
        backup(tmp_path, "bucket", "region", cloud)
    assert cloud.uploads == []


def test_backup_excludes_incomplete_directories_and_secrets(tmp_path):
    for name in (
        "runs/result.json",
        "runs/work.partial/result.json",
        "runs/.work/x",
        "runs/file.partial",
        "runs/file.lock",
        "configs/local.json",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    cloud = FakeS3()
    backup(tmp_path, "bucket", "region", cloud)
    assert set(json.loads(cloud.objects["latest.json"])["files"]) == {"runs/result.json"}


def test_backup_rejects_escaping_directory_symlink(tmp_path):
    root = tmp_path / "project"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret").write_text("not a project artifact")
    (root / "runs").symlink_to(outside, target_is_directory=True)
    cloud = FakeS3()
    with pytest.raises(ValueError, match="escapes project"):
        backup(root, "bucket", "region", cloud)
    assert cloud.uploads == []


def test_changed_immutable_artifact_cannot_publish_mixed_snapshot(tmp_path):
    path = tmp_path / "runs/result.json"
    path.parent.mkdir()
    path.write_text("first")
    cloud = FakeS3()
    backup(tmp_path, "bucket", "region", cloud)
    previous = cloud.objects["latest.json"]
    path.write_text("second")
    cloud.after_upload = lambda key: path.write_text("third")
    with pytest.raises(RuntimeError, match="Workspace changed"):
        backup(tmp_path, "bucket", "region", cloud)
    assert cloud.objects["latest.json"] == previous


def test_append_only_progress_log_can_continue_during_backup(tmp_path):
    path = tmp_path / "runs/events.jsonl"
    path.parent.mkdir()
    path.write_text('{"event": "started"}\n')
    captured = path.read_bytes()
    cloud = FakeS3()

    def append(key):
        if key.startswith("objects/"):
            with path.open("a") as stream:
                stream.write('{"event": "heartbeat"}\n')

    cloud.after_upload = append
    backup(tmp_path, "bucket", "region", cloud)
    manifest = json.loads(cloud.objects["latest.json"])
    assert manifest["files"]["runs/events.jsonl"] == hashlib.sha256(captured).hexdigest()
    assert path.read_bytes().startswith(captured)


@pytest.mark.parametrize("code", ["ConditionalRequestConflict", "AccessDenied"])
def test_failed_manifest_commit_never_reports_success(tmp_path, code):
    class FailedCommitS3(FakeS3):
        def put_object(self, **kwargs):
            raise ClientError({"Error": {"Code": code}}, "PutObject")

    (tmp_path / "runs").mkdir()
    (tmp_path / "runs/x").write_text("work")
    cloud = FailedCommitS3()
    exception = RuntimeError if code == "ConditionalRequestConflict" else ClientError
    with pytest.raises(exception):
        backup(tmp_path, "bucket", "region", cloud)
    assert "latest.json" not in cloud.objects
    events = [
        json.loads(line)["event"]
        for line in (tmp_path / "logs/cloud.jsonl").read_text().splitlines()
    ]
    assert "snapshot_committed" not in events
