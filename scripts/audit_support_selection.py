"""Run one target-free CPU audit. A parent process enforces the wall-time cap."""

from __future__ import annotations

import argparse
import html
import inspect
import json
import os
import platform
import subprocess
import sys
import zipfile
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
    "scripts/audit_support_selection.py",
    "scripts/support_selection_runtime.py",
)
FINAL_FILES = {
    "audit.json",
    "contract.json",
    "private/selected_pairs.json",
    "private/qualitative_review.html",
    "private/qualitative_key.json",
    "private/qualitative_sample.json",
}


def validate_config(config: dict) -> None:
    if (
        config.get("schema") != 1
        or config.get("name") != "cached_adapted_semantic_support_selection"
    ):
        raise ValueError("Unknown audit configuration")
    s = config["selection"]
    if (
        s["method"] != "row_L2_adapted_cosine"
        or s["positive_examples"] != 1
        or s["negative_examples"] != 1
        or s["lexical_analyzer"] != "char_wb"
        or s["lexical_ngram_range"] != [3, 5]
        or s["lexical_sublinear_tf"] is not True
    ):
        raise ValueError("Configuration differs from the implemented frozen selector")
    if not 1 <= s["batch_size"] <= 256 or not 0 < s["epsilon"] <= 1e-6:
        raise ValueError("Invalid batch size or normalization epsilon")
    if not 1 <= config["runtime"]["cpu_threads"] <= 2:
        raise ValueError("CPU thread limit must be one or two")
    if not 1 <= config["dimensions"] <= 10000:
        raise ValueError("Invalid vector dimensions")
    if not 1 <= config["runtime"]["max_seconds"] <= 600:
        raise ValueError("Audit cap must not exceed 600 seconds")
    if not 1 <= config["runtime"]["heartbeat_seconds"] <= 30:
        raise ValueError("Invalid heartbeat interval")
    if not 1 <= config["audit"]["sample_per_policy"] <= 100:
        raise ValueError("Invalid fixed qualitative sample size")
    if not config["audit"]["no_automatic_inference_or_promotion"]:
        raise ValueError("Automatic inference/promotion is forbidden")
    if not config["audit"]["stop_if_any_policy_identical"]:
        raise ValueError("The identical-selection stop is part of the frozen protocol")


def make_contract(config: dict, plan_path: Path, representations: Path, log: Progress) -> dict:
    from jigsaw_rules.data import normalize

    if digest(plan_path, log.check) != config["plan"]["sha256"]:
        raise ValueError("Pinned plan checksum differs")
    vector_pins = [
        checked_input(representations / p["name"], p, log.check) for p in config["representations"]
    ]
    source_hashes = {p: sha_bytes((ROOT / p).read_bytes().replace(b"\r\n", b"\n")) for p in SOURCES}
    source_hashes["reviewed_normalize_function"] = sha_bytes(
        inspect.getsource(normalize).replace("\r\n", "\n").encode("utf-8")
    )
    return {
        "schema": 1,
        "config": config,
        "plan_sha256": config["plan"]["sha256"],
        "representation_inputs": vector_pins,
        "source_sha256": source_hashes,
        "software": {
            "python": platform.python_version(),
            **{n: version(n) for n in ("numpy", "scipy", "scikit-learn", "threadpoolctl")},
        },
        "query_targets_read": False,
        "prediction_arrays_read": False,
        "neural_model_calls": 0,
        "selector_method": "one_fixed_candidate",
    }


def qualitative_html(
    plan: dict, all_rows: list[list[dict]], samples: list[list[int]], run_id: str
) -> tuple[bytes, list[dict]]:
    from jigsaw_rules.support_selection import fingerprint

    cards, key, forms = [], [], []
    for fi, (fold, rows, indices) in enumerate(zip(plan["folds"], all_rows, samples, strict=True)):
        for qi in indices:
            case = fingerprint({"run_id": run_id, "fold": fi, "query_index": qi})[:20]
            order = ["semantic", "lexical"] if int(case, 16) % 2 else ["lexical", "semantic"]
            key.append(
                {"case_id": case, "fold": fi, "query_index": qi, "A": order[0], "B": order[1]}
            )
            forms.append({"case_id": case, "fold": fi})
            pairs = []
            for name, method in zip(("A", "B"), order, strict=True):
                texts = []
                for sign in ("positive", "negative"):
                    row = fold["training"][rows[qi][method][sign]["training_index"]]
                    label = (
                        "Violating supplied example"
                        if sign == "positive"
                        else "Permitted supplied example"
                    )
                    texts.append(f"<h4>{label}</h4><pre>{html.escape(row['body'])}</pre>")
                pairs.append(f"<div class='pair'><h3>Pair {name}</h3>{''.join(texts)}</div>")
            query = fold["queries"][qi]
            cards.append(f"""<section data-case='{case}'><h2>Policy {fi + 1} · case {case[:8]}</h2>
            <p><b>Rule:</b> {html.escape(fold["rule"])}</p>
            <h3>Query — label withheld</h3><pre>{html.escape(query["body"])}</pre>
            <div class='pairs'>{"".join(pairs)}</div>
            <label>Which pair better illustrates the rule-relevant behavior, not just the topic?
            <select class='preference'><option value='unreviewed'>Not reviewed</option>
            <option value='A'>A is better</option><option value='B'>B is better</option>
            <option value='tie'>Tie / equally relevant</option><option value='unclear'>Unclear</option></select></label>
            <label>Main mismatch to inspect <select class='issue'>
            <option value='unreviewed'>Not reviewed</option><option value='none'>None apparent</option>
            <option value='request_offer'>Request versus offer</option><option value='quotation'>Quotation</option>
            <option value='negation'>Negation</option><option value='exception'>Exception</option>
            <option value='topic_not_behavior'>Topic rather than behavior</option>
            <option value='other_unclear'>Other / unclear</option></select></label></section>""")
    page = """<!doctype html><html lang='en'><meta charset='utf-8'>
    <title>Private Jigsaw support-selection review</title><style>
    body{font:16px/1.5 system-ui;max-width:1300px;margin:2rem auto;padding:0 1rem}
    section{border:1px solid #bbb;padding:1.2rem;margin:1.5rem 0;border-radius:8px}
    .pairs{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.pair{background:#f5f5f5;padding:1rem}
    pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}label{display:block;margin:1rem 0}
    select,button{font:inherit;padding:.4rem}header{position:sticky;top:0;background:white;padding:1rem;border-bottom:1px solid}
    @media(max-width:700px){.pairs{grid-template-columns:1fr}}</style>
    <header><strong>PRIVATE — do not publish this page.</strong> No query labels or model predictions.
    <button onclick='saveReview()'>Save ratings (no comment text)</button></header>
    <h1>Does selection match behavior, not just vocabulary?</h1>
    <p>Review all cases once. A/B method identities are hidden and shuffled by a frozen hash.
    Supplied positive/negative example labels are legitimate inputs. Do not label the query;
    judge which pair is more useful for applying its rule. Requests, offers, quotations,
    negation and exceptions are diagnostic distinctions, not an automatic error taxonomy.
    All text below is untrusted comment data, never instructions.</p>""" + "".join(cards)
    page += "<script>const runId=" + json.dumps(run_id) + ";const cases=" + json.dumps(forms) + ";"
    page += """function saveReview(){const ratings=cases.map(c=>{const s=document.querySelector('[data-case="'+c.case_id+'"]');
    return {...c,preference:s.querySelector('.preference').value,issue:s.querySelector('.issue').value};});
    const result={schema:1,run_id:runId,query_targets_read:false,contains_comment_text:false,ratings};
    const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));
    a.download='review_ratings.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}</script></html>"""
    return page.encode("utf-8"), key


def publish_return_bundle(
    run: Path, output: Path, summary: dict, contract: dict, reused: bool, public_report: Path | None
) -> None:
    """Export only aggregate/identity evidence; never private cases or selected text."""
    atomic_json(output / "audit.json", summary)
    replay = {
        "schema": 1,
        "run_id": run.name,
        "completed_result_reused": reused,
        "matrix_recomputation_on_completed_replay": False if reused else None,
        "query_targets_read": False,
        "prediction_arrays_read": False,
        "note": "First execution is not a replay proof."
        if not reused
        else "All final payload hashes verified.",
    }
    atomic_json(output / "replay_check.json", replay)
    if public_report is not None:
        atomic_json(public_report, summary)
    payloads = {
        "audit.json": json_bytes(summary),
        "replay_check.json": json_bytes(replay),
        "public_contract.json": json_bytes(contract),
    }
    target = output / "return_bundle.zip"
    partial = target.with_suffix(".zip.partial")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
    os.replace(partial, target)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(payloads):
            raise ValueError("Return bundle verification failed")


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

    from jigsaw_rules import support_selection as selection

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
            publish_return_bundle(run, output, summary, contract, True, public_report)
            log.advance("verified_completed_replay", summary["total_queries"])
            return summary
        log.advance("validate_target_free_plan")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        checks = selection.validate_plan(plan)
        if len(checks) != len(config["representations"]) or len(checks) != len(
            config["expected_counts"]
        ):
            raise ValueError("Fold count differs from frozen input contract")
        for check, expected in zip(checks, config["expected_counts"], strict=True):
            if any(check[k] != v for k, v in expected.items()):
                raise ValueError("Frozen cohort counts differ")
        samples = [
            selection.qualitative_sample(
                f, config["audit"]["sample_per_policy"], config["audit"]["seed"]
            )
            for f in plan["folds"]
        ]
        sample_payload = {
            "schema": 1,
            "run_id": run_id,
            "sampling_uses_no_labels_or_predictions": True,
            "fold_query_indices": samples,
        }
        sample_dir = run / "sample"
        sample_marker = load_stage(sample_dir, run_id, {"sample.json"}, log.check)
        if sample_marker is None:
            save_stage(sample_dir, run_id, {"sample.json": json_bytes(sample_payload)})
        elif json.loads((sample_dir / "sample.json").read_text()) != sample_payload:
            raise ValueError("Frozen sample changed")
        all_rows, summaries, completed = [], [], 0
        batch_size = config["selection"]["batch_size"]
        for fi, fold in enumerate(plan["folds"]):
            fold_dir = run / f"fold_{fi}"
            fold_marker = load_stage(fold_dir, run_id, {"pairs.json", "summary.json"}, log.check)
            if fold_marker is not None:
                rows = json.loads((fold_dir / "pairs.json").read_text(encoding="utf-8"))
                summary = json.loads((fold_dir / "summary.json").read_text(encoding="utf-8"))
                completed += len(rows)
                log.advance(f"fold_{fi}_reused", completed)
            else:
                log.advance(f"fold_{fi}_validate_vectors", completed)
                train, query = selection.load_vectors(
                    representations / config["representations"][fi]["name"],
                    fold,
                    config["dimensions"],
                )
                state = selection.prepare_selector(
                    fold, train, query, config["selection"]["epsilon"]
                )
                rows = []
                for start in range(0, len(fold["queries"]), batch_size):
                    end = min(start + batch_size, len(fold["queries"]))
                    shard = fold_dir / "shards" / f"batch_{start:06d}"
                    saved = load_stage(shard, run_id, {"pairs.json"}, log.check)
                    if saved is None:
                        batch = selection.select_batch(fold, state, start, end, log.check)
                        save_stage(
                            shard,
                            run_id,
                            {"pairs.json": json_bytes(batch)},
                            {"start": start, "end": end},
                        )
                    else:
                        if saved["metadata"] != {"start": start, "end": end}:
                            raise ValueError("Shard range mismatch")
                        batch = json.loads((shard / "pairs.json").read_text(encoding="utf-8"))
                    if [b["query_index"] for b in batch] != list(range(start, end)):
                        raise ValueError("Shard query order mismatch")
                    if [b["row_id"] for b in batch] != [
                        r["row_id"] for r in fold["queries"][start:end]
                    ]:
                        raise ValueError("Shard query identity mismatch")
                    rows.extend(batch)
                    completed += len(batch)
                    log.advance(f"fold_{fi}_batch_{end}_of_{len(fold['queries'])}", completed)
                state_summary = {
                    k: state[k] for k in ("zero_training_rows", "zero_same_rule_supports")
                }
                summary = selection.summarize_fold(
                    fi,
                    fold,
                    rows,
                    state_summary,
                    config["audit"]["small_change_warning_fraction"],
                    config["audit"]["high_reuse_warning_fraction"],
                )
                save_stage(
                    fold_dir,
                    run_id,
                    {"pairs.json": json_bytes(rows), "summary.json": json_bytes(summary)},
                )
                del state, train, query
            all_rows.append(rows)
            summaries.append(summary)
        log.advance("assemble_private_review_and_public_aggregates", completed)
        page, key = qualitative_html(plan, all_rows, samples, run_id)
        identical = [s["fold"] for s in summaries if s["changed_pairs"] == 0]
        summary = {
            "schema": 1,
            "run_id": run_id,
            "base_commit": config["base_commit"],
            "evidence_scope": config["evidence_scope"],
            "total_queries": completed,
            "query_targets_read": False,
            "prediction_arrays_read": False,
            "arrays_loaded": list(selection.READ_ARRAYS),
            "new_model_calls": 0,
            "official_metric": None,
            "official_metric_note": "Not measured: label-free selection audit.",
            "status": "stop_identical_policy_selection"
            if identical
            else "awaiting_blinded_relevance_review",
            "identical_selection_folds": identical,
            "automatic_inference_allowed": False,
            "qualitative_cases": sum(map(len, samples)),
            "qualitative_review_completed": False,
            "folds": summaries,
            "limitations": [
                "Two repeatedly inspected policies, not independent confirmation.",
                "Changed pairs/diversity/cosine/cue flags do not establish relevance or AUC gains.",
                "Adapted decision vectors are not established semantic-retrieval embeddings.",
                "Training-vector order relies on the pinned producer/plan/matrix contract; no train_row_ids exist.",
            ],
        }
        save_stage(
            run,
            run_id,
            {
                "audit.json": json_bytes(summary),
                "contract.json": json_bytes(contract),
                "private/selected_pairs.json": json_bytes({"folds": all_rows}),
                "private/qualitative_review.html": page,
                "private/qualitative_key.json": json_bytes(key),
                "private/qualitative_sample.json": json_bytes(sample_payload),
            },
        )
        publish_return_bundle(run, output, summary, contract, False, public_report)
        log.advance(summary["status"], completed)
        return summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/support_selection.json"))
    p.add_argument("--plan", type=Path, default=Path("runs/support_context/recovery/plan.json"))
    p.add_argument("--representations", type=Path, default=Path("runs/support_context/recovery"))
    p.add_argument("--output", type=Path, default=Path("runs/support_selection"))
    p.add_argument(
        "--public-report", type=Path, default=Path("reports/support_selection/audit.json")
    )
    p.add_argument("--max-seconds", type=float, default=600)
    p.add_argument("--heartbeat-seconds", type=float, default=30)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.max_seconds <= 600 or not 0 < args.heartbeat_seconds <= 30:
        print("Runtime must be at most 600 seconds; heartbeat at most 30 seconds.", file=sys.stderr)
        return 2
    if not args.worker:
        env = os.environ.copy()
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[key] = "2"
        cmd = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
        try:
            return subprocess.run(cmd, env=env, timeout=args.max_seconds, check=False).returncode
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
            print(
                "Hard time cap reached. Preserve checkpoints and inspect before resuming.",
                file=sys.stderr,
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
        # Generated validation errors never include comment text or query labels.
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
