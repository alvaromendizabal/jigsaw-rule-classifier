"""Recover only three pinned inputs. Network access requires explicit --download.

This program performs read-only AWS CLI calls when manually requested. It never
starts a job, writes to S3, widens permissions, or loads a model.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.support_selection_runtime import (  # noqa: E402
    Progress,
    atomic_bytes,
    atomic_json,
    checked_input,
    digest,
    sha_bytes,
)


def extract_plan(source: Path, destination: Path, spec: dict) -> None:
    """Read one regular archive member into memory; never extract a source tree."""
    with tarfile.open(source, "r:gz") as archive:
        members = [m for m in archive.getmembers() if m.name == spec["member"]]
        if len(members) != 1 or not members[0].isfile():
            raise ValueError("Plan member is absent, duplicated, or not a regular file")
        member = members[0]
        if not 0 < member.size <= spec["max_bytes"]:
            raise ValueError("Plan archive member exceeds the declared size limit")
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError("Plan member cannot be read")
        with stream:
            payload = stream.read(spec["max_bytes"] + 1)
    if len(payload) != member.size or sha_bytes(payload) != spec["sha256"]:
        raise ValueError("Extracted plan checksum or length differs")
    if destination.is_symlink():
        raise ValueError("Refusing a symlink plan destination")
    if destination.exists():
        if destination.read_bytes() != payload:
            raise ValueError("Existing plan differs; preserve and investigate")
    else:
        atomic_bytes(destination, payload)


def aws_call(
    args: list[str], profile: str, config: dict, log: Progress, capture: bool = False
) -> str:
    log.check()
    remaining = log.max_seconds - (time.monotonic() - log.started)
    env = {**os.environ, "AWS_MAX_ATTEMPTS": "1", "AWS_PAGER": ""}
    command = [
        "aws",
        *args,
        "--profile",
        profile,
        "--region",
        config["region"],
        "--cli-connect-timeout",
        "10",
        "--cli-read-timeout",
        "60",
    ]
    result = subprocess.run(
        command,
        check=False,
        timeout=max(0.1, min(90, remaining)),
        env=env,
        capture_output=capture,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            "AWS read failed; no automatic retry. Inspect local CLI authentication/permissions."
        )
    return result.stdout if capture else ""


def recover(
    config_path: Path,
    output: Path,
    download: bool = False,
    profile: str | None = None,
    max_seconds: float = 300,
) -> dict:
    from filelock import FileLock

    from jigsaw_rules import support_selection as selection

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not 0 < max_seconds <= min(300, config["runtime"]["recovery_max_seconds"]):
        raise ValueError("Recovery must be bounded to at most 300 seconds")
    if download and (not profile or profile.startswith("YOUR_")):
        raise ValueError("Supply your existing authorized AWS profile with --aws-profile")
    pins = [config["source"], *config["representations"]]
    if len(pins) != 3 or len({p["name"] for p in pins}) != 3:
        raise ValueError("Recovery is restricted to the source and two vector files")
    for pin in pins:
        if Path(pin["name"]).name != pin["name"] or ":" in pin["name"] or "\\" in pin["name"]:
            raise ValueError("Unsafe input filename")
    output.mkdir(parents=True, exist_ok=True)
    with (
        FileLock(str(output / ".recovery.lock"), timeout=0),
        Progress(output / "recovery_events.jsonl", max_seconds, 30) as log,
    ):
        verified, missing = {}, []
        for pin in pins:
            path = output / pin["name"]
            if path.exists():
                verified[pin["name"]] = checked_input(path, pin, log.check)
            else:
                missing.append(pin)
        if missing and not download:
            raise FileNotFoundError(
                "Pinned inputs missing. Use the documented explicit --download recovery step."
            )
        if missing:
            log.advance("verify_aws_account_before_read")
            identity = json.loads(
                aws_call(
                    ["sts", "get-caller-identity", "--output", "json"],
                    profile,
                    config,
                    log,
                    capture=True,
                )
            )
            if identity.get("Account") != config["account"]:
                raise ValueError("AWS account differs; no private artifacts downloaded")
        for pin in missing:
            log.advance("download_" + pin["name"])
            destination = output / pin["name"]
            partial = output / (pin["name"] + ".partial")
            # A completed transfer interrupted before rename can be reused. An
            # incomplete transfer is preserved for diagnosis, never overwritten.
            if partial.exists():
                checked_input(partial, pin, log.check)
            else:
                aws_call(
                    [
                        "s3api",
                        "get-object",
                        "--bucket",
                        config["bucket"],
                        "--key",
                        pin["key"],
                        "--expected-bucket-owner",
                        config["account"],
                        str(partial),
                    ],
                    profile,
                    config,
                    log,
                )
                checked_input(partial, pin, log.check)
            os.replace(partial, destination)
            verified[pin["name"]] = checked_input(destination, pin, log.check)
        log.advance("extract_only_target_free_plan")
        plan_path = output / config["plan"]["name"]
        extract_plan(output / config["source"]["name"], plan_path, config["plan"])
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        checks = selection.validate_plan(plan)
        if len(checks) != len(config["expected_counts"]) or len(checks) != len(
            config["representations"]
        ):
            raise ValueError("Recovered fold count differs")
        for i, (fold, check, expected, pin) in enumerate(
            zip(
                plan["folds"],
                checks,
                config["expected_counts"],
                config["representations"],
                strict=True,
            )
        ):
            if any(check[k] != v for k, v in expected.items()):
                raise ValueError("Recovered cohort counts differ")
            log.advance(f"verify_fold_{i}_shape_order_and_supports")
            train, query = selection.load_vectors(output / pin["name"], fold, config["dimensions"])
            _, train_zero = selection.normalized_vectors(train, config["selection"]["epsilon"])
            _, query_zero = selection.normalized_vectors(query, config["selection"]["epsilon"])
            if query_zero.any():
                raise ValueError("Recovered zero-norm query vector")
            for label in (0, 1):
                eligible = [
                    j
                    for j in check["support_indices"]
                    if fold["training"][j]["rule_violation"] == label
                ]
                if train_zero[eligible].all():
                    raise ValueError("Recovered support class contains only zero vectors")
            check["train_shape"], check["query_shape"] = list(train.shape), list(query.shape)
            check["zero_training_vectors"] = int(train_zero.sum())
            del check["support_indices"]  # No row-level inventory in return manifest.
            del train, query
        result = {
            "schema": 1,
            "status": "recovery_verified",
            "base_commit": config["base_commit"],
            "evidence_scope": config["evidence_scope"],
            "files": verified,
            "plan_sha256": digest(plan_path),
            "folds": checks,
            "query_targets_read": False,
            "prediction_arrays_read": False,
            "arrays_loaded": list(selection.READ_ARRAYS),
            "new_model_calls": 0,
            "private_text_included": False,
            "source_tree_extracted": False,
        }
        atomic_json(output / "recovery_manifest.json", result)
        log.advance("recovery_verified")
        return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/support_selection.json"))
    p.add_argument("--output", type=Path, default=Path("runs/support_context/recovery"))
    p.add_argument(
        "--download", action="store_true", help="Manually authorize only pinned S3 reads"
    )
    p.add_argument("--aws-profile")
    p.add_argument("--max-seconds", type=float, default=300)
    args = p.parse_args()
    try:
        recover(args.config, args.output, args.download, args.aws_profile, args.max_seconds)
        return 0
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        RuntimeError,
        subprocess.TimeoutExpired,
        tarfile.TarError,
    ) as exc:
        atomic_json(
            args.output / "recovery_failure.json",
            {
                "status": "stopped",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "automatic_retry": False,
            },
        )
        print(f"Recovery stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
