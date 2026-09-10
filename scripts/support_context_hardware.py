"""Bounded CUDA and private-storage preflight; no model or query data is loaded."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import time
import urllib.request
from datetime import UTC, datetime

INPUT_SHA256 = "ae27bdd2b71c95dc22e10c14e85cfd640f88c65e34e047ae7b9ae8792129d4b8"


def probe_cuda():
    import torch

    if torch.__version__ != "2.10.0+cu130" or torch.version.cuda != "13.0":
        raise ValueError("Expected the retained model's Torch/CUDA versions")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA with BF16 is required")
    name = torch.cuda.get_device_name(0)
    if "L4" not in name or "L40" in name:
        raise ValueError("Expected a single L4 GPU")
    if torch.cuda.device_count() != 1:
        raise ValueError("Expected exactly one GPU")
    x = torch.arange(128, device="cuda", dtype=torch.bfloat16).reshape(16, 8)
    first, replay = x.T @ x, x.T @ x
    if not torch.isfinite(first).all() or not torch.equal(first, replay):
        raise ValueError("CUDA finite-value/replay check failed")
    torch.cuda.synchronize()
    return {
        "gpu_tested": True,
        "gpu": name,
        "gpu_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "bf16_replay_verified": True,
    }


def public_cuda_script():
    """Public-only job payload: no private metadata, storage URLs or credentials."""
    return (
        "import json,platform,time\nfrom datetime import UTC,datetime\n"
        + inspect.getsource(probe_cuda)
        + "\nstart=time.monotonic()\nr=probe_cuda()\n"
        + 'r.update(status="passed",model_loaded=False,query_data_loaded=False,'
        + "private_storage_accessed=False,python=platform.python_version(),"
        + "elapsed_seconds=time.monotonic()-start,completed_utc=datetime.now(UTC).isoformat())\n"
        + "print(json.dumps(r,sort_keys=True),flush=True)\n"
    )


def read_url(name):
    with urllib.request.urlopen(os.environ[name], timeout=30) as response:
        return response.read(65536)


def run(storage_only=False):
    started = time.monotonic()
    required = ("JIGSAW_PROBE_INPUT", "JIGSAW_PROBE_PUT", "JIGSAW_PROBE_GET")
    present = {name: bool(os.environ.get(name)) for name in required}
    print(json.dumps({"phase": "secret_presence", "present": present}), flush=True)
    if not all(present.values()):
        raise RuntimeError("Required scoped storage secrets are missing")
    report = {
        "started_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "gpu_tested": False,
        "model_loaded": False,
        "query_data_loaded": False,
        "status": "failed",
    }
    try:
        metadata = read_url("JIGSAW_PROBE_INPUT")
        if hashlib.sha256(metadata).hexdigest() != INPUT_SHA256:
            raise ValueError("Pinned metadata checksum differs")
        report["input_sha256"] = INPUT_SHA256
        if not storage_only:
            report.update(probe_cuda())
        report["status"] = "passed"
    except Exception as error:
        # Never expose a signed URL or its credentials through exception details.
        report["error_type"] = type(error).__name__
    report["elapsed_seconds"] = time.monotonic() - started
    report["completed_utc"] = datetime.now(UTC).isoformat()
    payload = (json.dumps(report, sort_keys=True) + "\n").encode()
    request = urllib.request.Request(
        os.environ["JIGSAW_PROBE_PUT"],
        data=payload,
        method="PUT",
        headers={"x-amz-server-side-encryption": "AES256"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError("Private storage upload failed")
    if read_url("JIGSAW_PROBE_GET") != payload:
        raise ValueError("Private storage round-trip changed bytes")
    receipt = {
        **report,
        "storage_round_trip_verified": True,
        "stored_sha256": hashlib.sha256(payload).hexdigest(),
    }
    print(json.dumps(receipt, sort_keys=True), flush=True)
    if report["status"] != "passed":
        raise RuntimeError("Preflight failed; diagnostic is preserved in private storage")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--storage-only", action="store_true")
    modes.add_argument("--combined", action="store_true")
    modes.add_argument("--print-public-cuda-script", action="store_true")
    args = parser.parse_args()
    if args.print_public_cuda_script:
        print(public_cuda_script(), end="")
    else:
        run(args.storage_only)


if __name__ == "__main__":
    main()
