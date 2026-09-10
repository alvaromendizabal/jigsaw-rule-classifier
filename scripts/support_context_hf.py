"""Bounded HF entrypoint: authorized inputs, pinned runtime, durable outputs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path("/tmp/jigsaw/code")
PUBLIC_FILES = (
    "scripts/bootstrap_gpu.py",
    "scripts/support_context.py",
    "scripts/support_context_storage.py",
    "scripts/support_context_hf.py",
    "reports/checkpoints/support_context_transfer.json",
)
PACKAGES = (
    "transformers==4.57.6 huggingface-hub==0.36.2 accelerate==1.12.0 peft==0.18.0 "
    "numpy==2.3.5 pandas==2.2.3 scipy==1.17.0 scikit-learn==1.8.0 "
    "filelock==3.32.5 joblib==1.5.3 tokenizers==0.22.2 regex==2026.9.10"
).split()


def main():
    inputs = json.loads(os.environ["JIGSAW_INPUTS"])
    deadline = float(os.environ["JIGSAW_URL_DEADLINE"])
    if "--worker" not in sys.argv:
        commit = os.environ["JIGSAW_CODE_COMMIT"]
        if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise ValueError("A full immutable public code commit is required")
        ROOT.mkdir(parents=True, exist_ok=True)
        # Public code only; these URLs contain no private credentials.
        for name in PUBLIC_FILES:
            url = f"https://raw.githubusercontent.com/alvaromendizabal/jigsaw-rule-classifier/{commit}/{name}"
            target = ROOT / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=30) as response:
                target.write_bytes(response.read())
        sys.path.insert(0, str(ROOT))
        from scripts.support_context_storage import download

        manifest = json.loads((ROOT / PUBLIC_FILES[-1]).read_text())
        if manifest["status"] != "authorized_by_user" or len(inputs) != 6:
            raise ValueError("Explicit six-artifact authorization is required")
        for actual, expected in zip(inputs, manifest["inputs"], strict=True):
            if {k: v for k, v in actual.items() if k != "url"} != expected:
                raise ValueError("Private input scope differs from authorization")
        archive_path = ROOT.parent / "source.tar.gz"
        download(inputs[0]["url"], archive_path, inputs[0]["sha256"])
        # Preserve the public transport changes over the historical source bundle.
        public = {n: (ROOT / n).read_bytes() for n in PUBLIC_FILES}
        with tarfile.open(archive_path) as archive:
            archive.extractall(ROOT, filter="data")
        for name, payload in public.items():
            (ROOT / name).parent.mkdir(parents=True, exist_ok=True)
            (ROOT / name).write_bytes(payload)
        print(
            json.dumps(
                {
                    "stage": "source_verified",
                    "sha256": inputs[0]["sha256"],
                    "public_commit": commit,
                    "utc_epoch": time.time(),
                }
            ),
            flush=True,
        )
        wheel = ROOT.parent / "pip-25.3-py3-none-any.whl"
        download(
            "https://files.pythonhosted.org/packages/44/3c/"
            "d717024885424591d5376220b5e836c2d5293ce2011523c9de23ff7bf068/"
            "pip-25.3-py3-none-any.whl",
            wheel,
            "9655943313a94722b7774661c21049070f6bbb0a1516bf02f7c8d5d9201514cd",
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/bootstrap_gpu.py"),
                "/tmp/jigsaw/env",
                "--pip-wheel",
                str(wheel),
            ],
            check=True,
            timeout=120,
        )
        python = "/tmp/jigsaw/env/bin/python"
        subprocess.run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--constraint",
                "/tmp/jigsaw/env/native-constraints.txt",
                *PACKAGES,
            ],
            check=True,
            timeout=180,
        )
        subprocess.run([python, "-m", "pip", "check"], check=True, timeout=30)
        os.environ["PYTHONPATH"] = str(ROOT / "src") + ":" + str(ROOT)
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.chdir(ROOT)
        os.execv(python, [python, "-m", "scripts.support_context_hf", "--worker"])

    from scripts.support_context import run_worker
    from scripts.support_context_storage import SignedCheckpointStore, download

    bucket = "sagemaker-jigsaw-rules-560403859723-us-west-2"
    directory = ROOT.parent / "storage"
    store = SignedCheckpointStore(
        bucket,
        inputs,
        os.environ["JIGSAW_SNAPSHOT_PUT"],
        os.environ["JIGSAW_SNAPSHOT_GET"],
        deadline,
        directory,
    )
    # Verify the retained metadata too; the other inputs are checked as loaded.
    store.download_file(bucket, inputs[1]["key"], str(directory / "baseline_contract.json"))
    try:
        run_worker(
            ROOT,
            ROOT.parent / "work",
            bucket,
            "experiments/support-context-20260910",
            storage=store,
        )
    finally:
        store.flush()
    download(store.get_url, directory / "verified.tar.gz", store.last_snapshot_sha256)
    print(
        json.dumps({"stage": "FINAL_SNAPSHOT_VERIFIED", "sha256": store.last_snapshot_sha256}),
        flush=True,
    )


if __name__ == "__main__":
    main()
