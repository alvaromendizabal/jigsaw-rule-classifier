"""Bounded SageMaker CPU entrypoint for the registered asymmetric support audit.

This worker downloads only the hash-pinned target-free plan and cached adapted vectors,
executes no model calls, reads no query targets/predictions, and writes aggregate public
results plus a private fixed relevance-review sample. It is safe to replay because every
input and output is checksum bound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

import boto3
import numpy as np

CODE_ROOT = Path("/opt/ml/processing/input/code")
if CODE_ROOT.is_dir():
    sys.path.insert(0, str(CODE_ROOT))

from jigsaw_rules import asymmetric_support as candidate  # noqa: E402
from jigsaw_rules import support_selection as base  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(canonical_json(value))
    os.replace(temporary, path)


def emit(path: Path, event: str, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"utc": datetime.now(UTC).isoformat(), "event": event, **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)


def download_checked(s3, bucket: str, pin: dict, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, pin["key"], str(destination))
    size = destination.stat().st_size
    digest = sha256(destination)
    if size != pin["bytes"] or digest != pin["sha256"]:
        raise ValueError(f"Downloaded input differs: {pin['name']}")
    return {"name": pin["name"], "bytes": size, "sha256": digest, "key": pin["key"]}


def extract_plan(source: Path, pin: dict, destination: Path) -> dict:
    with tarfile.open(source, "r:gz") as archive:
        member = archive.getmember(pin["member"])
        if not member.isfile() or member.size > pin["max_bytes"]:
            raise ValueError("Plan member is not a bounded regular file")
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError("Plan member cannot be read")
        payload = handle.read(pin["max_bytes"] + 1)
    if len(payload) > pin["max_bytes"] or hashlib.sha256(payload).hexdigest() != pin["sha256"]:
        raise ValueError("Plan bytes differ from frozen checksum")
    destination.write_bytes(payload)
    plan = json.loads(payload)
    base.validate_plan(plan)
    return plan


def source_hashes(code_root: Path) -> dict[str, str]:
    names = (
        "jigsaw_rules/data.py",
        "jigsaw_rules/support_selection.py",
        "jigsaw_rules/asymmetric_support.py",
        "asymmetric_support_processing.py",
        "asymmetric_support.json",
    )
    result = {}
    for name in names:
        path = code_root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing processing source: {name}")
        result[name] = sha256(path)
    return result


def fixed_review_cases(fold: dict, rows: list[dict], *, sample_size: int, seed: int, fold_index: int) -> list[dict]:
    """Choose a deterministic target-blind sample and retain private text only in S3."""
    if sample_size < 1 or sample_size > len(rows):
        raise ValueError("Review sample size is outside the query cohort")

    def key(record: dict) -> str:
        payload = f"{seed}:{fold_index}:{record['row_id']}".encode()
        return hashlib.sha256(payload).hexdigest()

    chosen = sorted(rows, key=lambda record: (key(record), record["row_id"]))[:sample_size]
    cases = []
    for record in chosen:
        query = fold["queries"][record["query_index"]]
        case = {
            "case_id": hashlib.sha256(f"{fold_index}:{record['row_id']}".encode()).hexdigest()[:20],
            "fold": fold_index,
            "row_id": record["row_id"],
            "rule": fold["rule"],
            "query_body": query["body"],
            "query_target_included": False,
            "methods": {},
        }
        for method in ("lexical", "semantic", "asymmetric"):
            pair = {}
            for sign in ("positive", "negative"):
                selected = record[method][sign]
                index = selected["training_index"]
                row = fold["training"][index]
                pair[sign] = {
                    "training_index": index,
                    "body": row["body"],
                    "label": row["rule_violation"],
                    "support_id": selected["support_id"],
                    "cosine": selected["cosine"],
                }
            case["methods"][method] = pair
        cases.append(case)
    return cases


def run(config_path: Path, output: Path, work: Path) -> dict:
    started = time.monotonic()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != 1 or config.get("name") != "asymmetric_cached_support_selection":
        raise ValueError("Unexpected processing config")
    if config["runtime"]["max_seconds"] > 600 or config["runtime"]["cpu_threads"] not in (1, 2):
        raise ValueError("Processing runtime contract is too broad")
    if not config["audit"]["no_automatic_inference_or_promotion"]:
        raise ValueError("Automatic inference/promotion must remain disabled")

    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(config["runtime"]["cpu_threads"])

    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    events = output / "events.jsonl"
    emit(events, "start", base_commit=config["base_commit"], model_calls=0, query_targets_read=False)
    s3 = boto3.client("s3", region_name=config["region"])

    source_path = work / config["source"]["name"]
    input_records = [download_checked(s3, config["bucket"], config["source"], source_path)]
    vector_paths = []
    for pin in config["representations"]:
        path = work / pin["name"]
        input_records.append(download_checked(s3, config["bucket"], pin, path))
        vector_paths.append(path)
    emit(events, "inputs_verified", files=len(input_records))

    plan_path = work / config["plan"]["name"]
    plan = extract_plan(source_path, config["plan"], plan_path)
    checks = base.validate_plan(plan)
    if len(checks) != len(config["expected_counts"]):
        raise ValueError("Fold count differs from frozen config")
    for actual, expected in zip(checks, config["expected_counts"], strict=True):
        for key, expected_value in expected.items():
            if actual[key] != expected_value:
                raise ValueError(f"Frozen cohort count differs: {key}")
    emit(events, "plan_verified", folds=len(checks))

    all_rows, summaries, review_cases = [], [], []
    batch_size = config["selection"]["batch_size"]
    for fi, (fold, vector_path) in enumerate(zip(plan["folds"], vector_paths, strict=True)):
        train, query = base.load_vectors(vector_path, fold, config["dimensions"])
        state = base.prepare_selector(fold, train, query, config["selection"]["epsilon"])
        rows = []
        for start in range(0, len(fold["queries"]), batch_size):
            end = min(start + batch_size, len(fold["queries"]))
            rows.extend(candidate.select_batch(fold, state, start, end))
            emit(events, "batch_complete", fold=fi, end=end, total=len(fold["queries"]))
        summary = candidate.summarize_fold(
            fi,
            fold,
            rows,
            {
                "zero_training_rows": state["zero_training_rows"],
                "zero_same_rule_supports": state["zero_same_rule_supports"],
            },
        )
        if summary["queries"] != len(fold["queries"]):
            raise ValueError("Fold result is incomplete")
        all_rows.append(rows)
        summaries.append(summary)
        review_cases.extend(
            fixed_review_cases(
                fold,
                rows,
                sample_size=config["audit"]["sample_per_policy"],
                seed=config["audit"]["seed"],
                fold_index=fi,
            )
        )
        emit(events, "fold_complete", fold=fi, queries=len(rows))

    identical_lexical = [s["fold"] for s in summaries if s["changed_vs_lexical"] == 0]
    status = "stop_policy_matches_lexical" if identical_lexical else "awaiting_assistant_blinded_review"
    contract = {
        "schema": 1,
        "base_commit": config["base_commit"],
        "config_sha256": sha256(config_path),
        "plan_sha256": config["plan"]["sha256"],
        "inputs": input_records,
        "source_sha256": source_hashes(CODE_ROOT if CODE_ROOT.is_dir() else config_path.parent),
        "query_targets_read": False,
        "prediction_arrays_read": False,
        "model_calls": 0,
    }
    audit = {
        "schema": 1,
        "method": candidate.METHOD,
        "status": status,
        "total_queries": sum(s["queries"] for s in summaries),
        "query_targets_read": False,
        "prediction_arrays_read": False,
        "model_calls": 0,
        "official_metric": None,
        "official_metric_note": "Not measured: target-free cached-vector selector audit.",
        "identical_to_lexical_folds": identical_lexical,
        "folds": summaries,
        "elapsed_seconds": time.monotonic() - started,
        "limitations": [
            "Two repeatedly inspected policies; not independent confirmation.",
            "Selection diagnostics do not establish relevance or AUC improvement.",
            "Farthest permitted supports may be irrelevant; blinded review is required before inference.",
        ],
    }
    atomic_json(output / "contract.json", contract)
    atomic_json(output / "audit.json", audit)
    atomic_json(output / "private" / "selected_pairs.json", {"folds": all_rows})
    atomic_json(output / "private" / "review_cases.json", {"cases": review_cases})
    files = {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "complete.json"
    }
    atomic_json(
        output / "complete.json",
        {"schema": 1, "status": status, "files": files, "completed_utc": datetime.now(UTC).isoformat()},
    )
    emit(events, "complete", status=status, queries=audit["total_queries"])
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CODE_ROOT / "asymmetric_support.json")
    parser.add_argument("--output", type=Path, default=Path("/opt/ml/processing/output"))
    parser.add_argument("--work", type=Path, default=Path("/opt/ml/processing/work"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run(args.config, args.output, args.work)
        return 0
    except Exception as exc:  # failure receipt is intentional; SageMaker still gets a nonzero job.
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(
            args.output / "failure.json",
            {"status": "stopped", "error_type": type(exc).__name__, "message": str(exc), "automatic_retry": False},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
