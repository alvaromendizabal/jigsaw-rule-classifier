"""Run one bounded target-free audit of asymmetric cached-vector support selection."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.support_selection_runtime import (  # noqa: E402
    Progress,
    atomic_json,
    checked_input,
    digest,
    json_bytes,
    load_stage,
    save_stage,
    sha_bytes,
)

SOURCES = (
    "src/jigsaw_rules/support_selection.py",
    "src/jigsaw_rules/asymmetric_support.py",
    "scripts/audit_asymmetric_support.py",
    "scripts/support_selection_runtime.py",
)
FINAL_FILES = {"audit.json", "contract.json", "private/selected_pairs.json"}


def validate_config(config: dict) -> None:
    if config.get("schema") != 1 or config.get("name") != "asymmetric_cached_support_selection":
        raise ValueError("Unknown asymmetric-selection configuration")
    selection = config["selection"]
    if (
        selection["method"] != "nearest_positive_farthest_negative"
        or selection["positive_examples"] != 1
        or selection["negative_examples"] != 1
    ):
        raise ValueError("Configuration differs from the registered asymmetric selector")
    if not 0 < selection["epsilon"] <= 1e-6 or not 1 <= selection["batch_size"] <= 256:
        raise ValueError("Invalid selection numeric bounds")
    runtime = config["runtime"]
    if not 1 <= runtime["cpu_threads"] <= 2:
        raise ValueError("CPU thread limit must be one or two")
    if not 1 <= runtime["max_seconds"] <= 600:
        raise ValueError("Audit cap must not exceed 600 seconds")
    if not 1 <= runtime["heartbeat_seconds"] <= 30:
        raise ValueError("Heartbeat must not exceed 30 seconds")
    if not config["audit"]["no_automatic_inference_or_promotion"]:
        raise ValueError("Automatic inference/promotion is forbidden")
    if not config["audit"]["stop_if_any_policy_matches_lexical"]:
        raise ValueError("The registered identical-policy stop is required")


def make_contract(config: dict, plan_path: Path, representations: Path, log: Progress) -> dict:
    from jigsaw_rules.data import normalize

    if digest(plan_path, log.check) != config["plan"]["sha256"]:
        raise ValueError("Pinned plan checksum differs")
    vector_pins = [
        checked_input(representations / pin["name"], pin, log.check)
        for pin in config["representations"]
    ]
    sources = {
        path: sha_bytes((ROOT / path).read_bytes().replace(b"\r\n", b"\n"))
        for path in SOURCES
    }
    sources["reviewed_normalize_function"] = sha_bytes(
        inspect.getsource(normalize).replace("\r\n", "\n").encode("utf-8")
    )
    return {
        "schema": 1,
        "config": config,
        "plan_sha256": config["plan"]["sha256"],
        "representation_inputs": vector_pins,
        "source_sha256": sources,
        "software": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl")},
        },
        "query_targets_read": False,
        "prediction_arrays_read": False,
        "neural_model_calls": 0,
        "selector_method": "one_frozen_asymmetric_candidate",
    }


def run_audit(
    config_path: Path,
    plan_path: Path,
    representations: Path,
    output: Path,
    max_seconds: float = 600,
    heartbeat_seconds: float = 30,
    public_report: Path | None = None,
) -> dict:
    from filelock import FileLock
    from threadpoolctl import threadpool_limits

    from jigsaw_rules import asymmetric_support as candidate
    from jigsaw_rules import support_selection as base

    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    if not 0 < max_seconds <= config["runtime"]["max_seconds"]:
        raise ValueError("Requested runtime exceeds the frozen cap")
    if not 0 < heartbeat_seconds <= 30:
        raise ValueError("Heartbeat must be within 30 seconds")
    output.mkdir(parents=True, exist_ok=True)
    with (
        FileLock(str(output / ".audit.lock"), timeout=0),
        Progress(output / "events.jsonl", max_seconds, heartbeat_seconds) as log,
        threadpool_limits(limits=config["runtime"]["cpu_threads"]),
    ):
        log.advance("verify_inputs_and_source")
        contract = make_contract(config, plan_path, representations, log)
        run_id = sha_bytes(json_bytes(contract))[:24]
        run = output / run_id
        marker = load_stage(run, run_id, FINAL_FILES, log.check)
        if marker is not None:
            summary = json.loads((run / "audit.json").read_text(encoding="utf-8"))
            if public_report is not None:
                atomic_json(public_report, summary)
            log.advance("verified_completed_replay", summary["total_queries"])
            return summary

        log.advance("validate_target_free_plan")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        checks = base.validate_plan(plan)
        if len(checks) != len(config["representations"]):
            raise ValueError("Fold count differs from frozen input contract")
        for check, expected in zip(checks, config["expected_counts"], strict=True):
            for key, value in expected.items():
                if check[key] != value:
                    raise ValueError("Frozen cohort counts differ")

        all_rows, summaries, completed = [], [], 0
        batch_size = config["selection"]["batch_size"]
        for fi, fold in enumerate(plan["folds"]):
            fold_dir = run / f"fold_{fi}"
            saved = load_stage(fold_dir, run_id, {"pairs.json", "summary.json"}, log.check)
            if saved is not None:
                rows = json.loads((fold_dir / "pairs.json").read_text(encoding="utf-8"))
                summary = json.loads((fold_dir / "summary.json").read_text(encoding="utf-8"))
                completed += len(rows)
                log.advance(f"fold_{fi}_reused", completed)
            else:
                log.advance(f"fold_{fi}_validate_vectors", completed)
                train, query = base.load_vectors(
                    representations / config["representations"][fi]["name"],
                    fold,
                    config["dimensions"],
                )
                state = base.prepare_selector(fold, train, query, config["selection"]["epsilon"])
                rows = []
                for start in range(0, len(fold["queries"]), batch_size):
                    end = min(start + batch_size, len(fold["queries"]))
                    shard = fold_dir / "shards" / f"batch_{start:06d}"
                    shard_record = load_stage(shard, run_id, {"pairs.json"}, log.check)
                    if shard_record is None:
                        batch = candidate.select_batch(fold, state, start, end, log.check)
                        save_stage(
                            shard,
                            run_id,
                            {"pairs.json": json_bytes(batch)},
                            {"start": start, "end": end},
                        )
                    else:
                        if shard_record["metadata"] != {"start": start, "end": end}:
                            raise ValueError("Shard range mismatch")
                        batch = json.loads((shard / "pairs.json").read_text(encoding="utf-8"))
                    if [row["query_index"] for row in batch] != list(range(start, end)):
                        raise ValueError("Shard query order mismatch")
                    if [row["row_id"] for row in batch] != [
                        row["row_id"] for row in fold["queries"][start:end]
                    ]:
                        raise ValueError("Shard query identity mismatch")
                    rows.extend(batch)
                    completed += len(batch)
                    log.advance(f"fold_{fi}_batch_{end}_of_{len(fold['queries'])}", completed)
                summary = candidate.summarize_fold(
                    fi,
                    fold,
                    rows,
                    {
                        "zero_training_rows": state["zero_training_rows"],
                        "zero_same_rule_supports": state["zero_same_rule_supports"],
                    },
                )
                save_stage(
                    fold_dir,
                    run_id,
                    {"pairs.json": json_bytes(rows), "summary.json": json_bytes(summary)},
                )
                del state, train, query
            all_rows.append(rows)
            summaries.append(summary)

        identical_lexical = [
            item["fold"] for item in summaries if item["changed_vs_lexical"] == 0
        ]
        identical_semantic = [
            item["fold"] for item in summaries if item["changed_vs_semantic"] == 0
        ]
        status = (
            "stop_policy_matches_lexical"
            if identical_lexical
            else "awaiting_assistant_blinded_relevance_review"
        )
        summary = {
            "schema": 1,
            "run_id": run_id,
            "base_commit": config["base_commit"],
            "evidence_scope": config["evidence_scope"],
            "method": candidate.METHOD,
            "total_queries": completed,
            "query_targets_read": False,
            "prediction_arrays_read": False,
            "arrays_loaded": list(base.READ_ARRAYS),
            "new_model_calls": 0,
            "official_metric": None,
            "official_metric_note": "Not measured: label-free selection audit.",
            "status": status,
            "identical_to_lexical_folds": identical_lexical,
            "identical_to_nearest_semantic_folds": identical_semantic,
            "automatic_inference_allowed": False,
            "folds": summaries,
            "limitations": [
                "Two repeatedly inspected policies, not independent confirmation.",
                "Selection diagnostics do not establish relevance or AUC gains.",
                "Farthest negatives can become irrelevant; qualitative review is mandatory "
                "before inference.",
                "Adapted decision vectors are not established retrieval embeddings.",
            ],
        }
        save_stage(
            run,
            run_id,
            {
                "audit.json": json_bytes(summary),
                "contract.json": json_bytes(contract),
                "private/selected_pairs.json": json_bytes({"folds": all_rows}),
            },
        )
        if public_report is not None:
            atomic_json(public_report, summary)
        log.advance(status, completed)
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/asymmetric_support.json"))
    parser.add_argument(
        "--plan", type=Path, default=Path("runs/support_context/recovery/plan.json")
    )
    parser.add_argument(
        "--representations", type=Path, default=Path("runs/support_context/recovery")
    )
    parser.add_argument("--output", type=Path, default=Path("runs/asymmetric_support"))
    parser.add_argument(
        "--public-report", type=Path, default=Path("reports/asymmetric_support/audit.json")
    )
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--heartbeat-seconds", type=float, default=30)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.max_seconds <= 600 or not 0 < args.heartbeat_seconds <= 30:
        print("Runtime must be at most 600 seconds; heartbeat at most 30 seconds.", file=sys.stderr)
        return 2
    if not args.worker:
        env = os.environ.copy()
        for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"
        ):
            env[key] = "2"
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
        try:
            completed = subprocess.run(
                command, env=env, timeout=args.max_seconds, check=False
            )
            return completed.returncode
        except subprocess.TimeoutExpired:
            args.output.mkdir(parents=True, exist_ok=True)
            atomic_json(
                args.output / "timeout.json",
                {
                    "status": "hard_wall_time_limit",
                    "max_seconds": args.max_seconds,
                    "completed_shards_preserved": True,
                    "automatic_retry": False,
                },
            )
            return 124
    try:
        run_audit(
            args.config,
            args.plan,
            args.representations,
            args.output,
            args.max_seconds,
            args.heartbeat_seconds,
            args.public_report,
        )
        return 0
    except (ValueError, OSError, KeyError, TypeError, TimeoutError) as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(
            args.output / "failure.json",
            {
                "status": "stopped",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "automatic_retry": False,
            },
        )
        print(f"Stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 124 if isinstance(exc, TimeoutError) else 2


if __name__ == "__main__":
    raise SystemExit(main())
