"""Compare verified cached feature signals with the Qwen development reference.

No fitting, new neural inference, model loading, or learned combination.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from jigsaw_rules.data import normalize, validate_frame
from jigsaw_rules.runtime import atomic_json, digest
from scripts.run_behavioral_features import protocol, weighted_auc_samples

PUBLIC = "reports/feature_value_audit"
PRIVATE = "runs/feature_value_audit"
CONFIG = "configs/feature_value_audit.json"
REFERENCE = "qwen_adapted_reference"
VARIANTS = (
    "lexical_control",
    "add_behavior",
    "add_act_roles",
    "condition_lexical",
    "rule_both",
    "copy_both",
    "scope_both",
)
SLICES = {
    "explicit_quotation": r'(?m)^\s*>|"[^"\n]{2,}"|“[^”\n]{2,}”',
    "code_markup": r"`",
    "negation_word": r"\b(?:not|never|no|without|cannot)\b|n't\b",
    "question_mark": r"\?",
    "link": r"https?://|\bwww\.",
    "advice_directive": r"\b(?:you should|you must|you need to|i recommend)\b",
    "first_person_request": r"\b(?:should i|can i|how do i|could i)\b",
}


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def probabilities(value, size):
    value = np.asarray(value, dtype=float)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError("probabilities must be finite and aligned")
    if ((value < 0) | (value > 1)).any():
        raise ValueError("probabilities outside [0, 1]")
    return value


def binary_labels(value):
    y = np.asarray(value)
    if y.ndim != 1 or not np.isin(y, [0, 1]).all():
        raise ValueError("binary labels required")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("both label classes required")
    return y.astype(int)


def ordered_ids(value, expected):
    ids = np.asarray(value)
    if ids.ndim != 1 or ids.dtype.kind not in "iu":
        raise ValueError("integer row IDs required")
    if len(np.unique(ids)) != len(ids) or not np.array_equal(ids, expected):
        raise ValueError("query IDs/order differ from the canonical cohort")
    return ids


def pair_decomposition(y, reference, candidate):
    """Exact ROC-AUC decomposition with half credit for ties.

    Positive and negative examples form pairs. These are dependent pairs;
    they must not be treated as independent statistical observations.
    """
    y = binary_labels(y)
    ref = probabilities(reference, len(y))
    alt = probabilities(candidate, len(y))
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(pos) * len(neg) > 2_000_000:
        raise ValueError("pair budget exceeded")

    def credit(p):
        difference = p[pos, None] - p[neg][None, :]
        return (difference > 0).astype(float) + 0.5 * (difference == 0)

    a, b = credit(ref), credit(alt)
    delta = b - a
    gained = float(np.maximum(delta, 0).mean())
    lost = float(np.maximum(-delta, 0).mean())
    net = float(roc_auc_score(y, alt) - roc_auc_score(y, ref))
    if not np.isclose(gained - lost, net, atol=1e-12, rtol=0):
        raise AssertionError("pairwise decomposition does not equal AUC difference")
    return {
        "pairs": int(delta.size),
        "gained_auc_credit": gained,
        "lost_auc_credit": lost,
        "net_auc_delta": net,
        "pairs_improved": int((delta > 0).sum()),
        "pairs_worsened": int((delta < 0).sum()),
        "reference_ties": int((a == 0.5).sum()),
        "candidate_ties": int((b == 0.5).sum()),
        "pair_count_is_not_independent_sample_size": True,
    }


def slice_flags(query):
    if set(query.columns) != {"row_id", "body", "rule"}:
        raise ValueError("slice construction requires a target-free query frame")
    return {
        name: query.body.str.contains(pattern, regex=True, flags=re.I).to_numpy()
        for name, pattern in SLICES.items()
    }


def rank_correlation(a, b):
    x, y = rankdata(a), rankdata(b)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def metrics(y, p):
    return {
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def summarize(folds, *, replicates=500, seed=20260917, min_slice_class=10):
    """All labels are used only here for retrospective scoring, never fitting."""
    models = (REFERENCE, *VARIANTS)
    if len(folds) != 2 or not isinstance(replicates, int) or replicates < 20:
        raise ValueError("two canonical folds and >=20 bootstrap draws required")
    groups = sorted({normalize(text) for fold in folds for text in fold["query"].body})
    lookup = {text: i for i, text in enumerate(groups)}
    if len(groups) > 5000 or replicates > 2000:
        raise ValueError("bootstrap memory budget exceeded")
    weights = np.random.default_rng(seed).multinomial(
        len(groups), np.full(len(groups), 1 / len(groups)), size=replicates
    )
    rows, pairs, slices, correlations = [], [], [], []
    samples = {name: [] for name in models}
    points = {name: [] for name in models}
    for index, fold in enumerate(folds):
        query = fold["query"]
        flags = slice_flags(query)
        y = binary_labels(fold["y"])
        if len(y) != len(query) or set(fold["predictions"]) != set(models):
            raise ValueError("model/label/query schema differs")
        w = weights[:, [lookup[normalize(text)] for text in query.body]]
        probs = {name: probabilities(fold["predictions"][name], len(y)) for name in models}
        for name, p in probs.items():
            score = metrics(y, fold.get("probabilities", probs)[name])
            score["auc"] = float(roc_auc_score(y, p))
            points[name].append(score["auc"])
            rows.append({"fold": index, "policy": fold["rule"], "model": name, **score})
            samples[name].append(weighted_auc_samples(y, p, w))
            if name != REFERENCE:
                pairs.append(
                    {
                        "fold": index,
                        "policy": fold["rule"],
                        "model": name,
                        **pair_decomposition(y, probs[REFERENCE], p),
                    }
                )
        for a in models:
            for b in models:
                correlations.append(
                    {
                        "fold": index,
                        "left": a,
                        "right": b,
                        "spearman": rank_correlation(probs[a], probs[b]),
                    }
                )
        # No tuning or ranking of slices. Every fixed slice and its complement is shown.
        for slice_name, mask in flags.items():
            for present in (True, False):
                part = mask if present else ~mask
                positives, negatives = int(y[part].sum()), int((1 - y[part]).sum())
                eligible = min(positives, negatives) >= min_slice_class
                for name in (REFERENCE, "rule_both", "scope_both"):
                    row = {
                        "fold": index,
                        "policy": fold["rule"],
                        "slice": slice_name,
                        "present": present,
                        "rows": int(part.sum()),
                        "positive_rows": positives,
                        "negative_rows": negatives,
                        "model": name,
                        "auc": None,
                        "status": "insufficient_class_support",
                    }
                    if eligible:
                        row.update(
                            auc=float(roc_auc_score(y[part], probs[name][part])),
                            status="descriptive_only",
                        )
                    slices.append(row)
    delta = np.array([np.mean(points[name]) - np.mean(points[REFERENCE]) for name in VARIANTS])
    draws = np.column_stack(
        [np.mean(samples[name], axis=0) - np.mean(samples[REFERENCE], axis=0) for name in VARIANTS]
    )
    valid = np.isfinite(draws).all(axis=1)
    if valid.sum() < 0.9 * replicates:
        raise ValueError("insufficient valid comment-group bootstrap draws")
    width = float(np.quantile(np.max(np.abs(draws[valid] - delta), axis=1), 0.95))
    contrasts = [
        {
            "model": name,
            "delta_auc": float(delta[i]),
            "simultaneous_low": float(delta[i] - width),
            "simultaneous_high": float(delta[i] + width),
            "valid_draws": int(valid.sum()),
        }
        for i, name in enumerate(VARIANTS)
    ]
    return {
        "metrics": rows,
        "pair_decomposition": pairs,
        "slices": slices,
        "correlations": correlations,
        "contrasts": contrasts,
        "macro_auc": {name: float(np.mean(points[name])) for name in models},
        "model_fits": 0,
        "new_neural_inference": 0,
        "blends_tested": 0,
        "automatic_gpu_authorization": False,
        "promotion": "NONE_DIAGNOSTIC_ONLY",
        "next_gate": "choose_one_preregistered_feature_hypothesis_after_review",
    }


def verified_checkpoint(folder, expected):
    marker = folder / "complete.json"
    if not marker.is_file():
        raise ValueError("missing completed prediction checkpoint; do not refit")
    record = json.loads(marker.read_text())
    if record.get("identity") != expected or not record.get("files"):
        raise ValueError("checkpoint identity mismatch")
    for name, checksum in record["files"].items():
        if Path(name).name != name or digest(folder / name) != checksum:
            raise ValueError("private prediction checkpoint checksum mismatch")
    if "predictions.npz" not in record["files"]:
        raise ValueError("prediction file missing from checkpoint manifest")
    return folder / "predictions.npz"


def load_rounds(root, config):
    """Verify all five report/source lineages without importing their test fixtures."""
    reports = {}
    for item in config["rounds"]:
        path = root / item["public"] / "results.json"
        if digest(path) != item["sha256"]:
            raise ValueError("completed round report differs: " + item["public"])
        result = json.loads(path.read_text())
        if result["run_id"] != item["run_id"]:
            raise ValueError("completed round identity differs")
        for name, checksum in result["identity"]["source"].items():
            if digest(root / name) != checksum:
                raise ValueError("historical source changed; preserve it: " + name)
        directory = root / item["private"] / item["run_id"]
        marker = json.loads((directory / "finished.json").read_text())
        # Round 1 uses the same identity but its completion layout is inspected separately.
        if marker.get("identity") != result["identity"]:
            raise ValueError("historical completion identity differs")
        for name, checksum in marker.get("public_hashes", {}).items():
            if Path(name).name != name or digest(root / item["public"] / name) != checksum:
                raise ValueError("historical aggregate checksum mismatch")
        reports[item["round"]] = (result, directory)
    return reports


def reference_candidates(root, home, item):
    """Search only relevant cache roots, filenames and exact expected byte sizes."""
    starts = [root / "runs", home / "jigsaw-preserved", home / "jigsaw-research-evidence"]
    starts += [home / "jigsaw-recovery", home / "jigsaw-audit"]
    names = {"representations.npz", item["name"]}
    visited = 0
    for start in starts:
        if not start.is_dir() or start.is_symlink():
            continue
        for parent, dirs, files in os.walk(start, followlinks=False):
            dirs[:] = [
                d
                for d in sorted(dirs)
                if d
                not in {
                    ".git",
                    ".venv",
                    "site-packages",
                    "model",
                    "models",
                    "node_modules",
                }
                and not (Path(parent) / d).is_symlink()
            ]
            visited += 1
            if visited > 6000:
                return
            for name in sorted(set(files) & names):
                path = Path(parent) / name
                if not path.is_symlink() and path.stat().st_size == item["bytes"]:
                    yield path


def atomic_copy_verified(source, target, item):
    if target.exists():
        raise ValueError("reference target appeared concurrently; preserve it")
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".partial", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with source.open("rb") as data:
                for block in iter(lambda: data.read(1024 * 1024), b""):
                    stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
            if temporary.stat().st_size != item["bytes"] or digest(temporary) != item["sha256"]:
                raise ValueError("reference copy checksum mismatch")
            # Exclusive hard-link publication cannot overwrite a concurrent target.
            os.link(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


def recover_reference(root, home, config, *, allow_s3, log, clients=None):
    cache = root / PRIVATE / "reference"
    cache.mkdir(parents=True, exist_ok=True)
    receipts, paths = [], []
    for index, item in enumerate(config["reference_files"]):
        target = cache / item["name"]
        if target.is_symlink():
            raise ValueError("symlink reference target not permitted")
        if target.exists():
            if target.stat().st_size != item["bytes"] or digest(target) != item["sha256"]:
                raise ValueError("saved reference is corrupt; preserve it, do not redownload")
            origin = "verified_local_cache"
        else:
            source = next(
                (p for p in reference_candidates(root, home, item) if digest(p) == item["sha256"]),
                None,
            )
            if source is not None:
                atomic_copy_verified(source, target, item)
                origin = "verified_existing_private_copy"
            else:
                if not allow_s3:
                    raise ValueError("reference absent; explicit --recover-reference required")
                if clients is None:
                    import boto3
                    from botocore.config import Config

                    options = Config(
                        connect_timeout=5, read_timeout=20, retries={"total_max_attempts": 1}
                    )
                    clients = (
                        boto3.client("sts", region_name=config["region"], config=options),
                        boto3.client("s3", region_name=config["region"], config=options),
                    )
                if clients[0].get_caller_identity()["Account"] != config["account"]:
                    raise ValueError("unexpected AWS account; no S3 read permitted")
                response = clients[1].get_object(Bucket=config["bucket"], Key=item["key"])
                body = response["Body"]
                try:
                    if response["ContentLength"] != item["bytes"]:
                        raise ValueError("S3 reference size differs")
                    with tempfile.NamedTemporaryFile(dir=cache, delete=False) as temp:
                        temporary = Path(temp.name)
                        try:
                            received = 0
                            while block := body.read(1024 * 1024):
                                received += len(block)
                                if received > item["bytes"]:
                                    raise ValueError("reference download byte budget exceeded")
                                temp.write(block)
                            temp.flush()
                            os.fsync(temp.fileno())
                            if received != item["bytes"] or digest(temporary) != item["sha256"]:
                                raise ValueError("download checksum/length mismatch")
                            os.link(temporary, target)
                        finally:
                            temporary.unlink(missing_ok=True)
                finally:
                    body.close()
                origin = "verified_private_s3_read"
        receipts.append(
            {"fold": index, "origin": origin, "bytes": item["bytes"], "sha256": item["sha256"]}
        )
        paths.append(target)
        atomic_json(cache / f"fold_{index}_receipt.json", receipts[-1])
        log.emit("reference_verified", fold=index, origin=origin)
    return paths, receipts


def load_folds(root, config, reports, paths):
    path = root / "data/raw/train.csv"
    if digest(path) != config["train_sha256"]:
        raise ValueError("original training checksum differs")
    frame = pd.read_csv(path)
    validate_frame(frame, train=True)
    plan, cohorts = protocol(frame, config["query_counts"])
    fifth = reports[5][0]
    if [c["query_identity"] for c in cohorts] != [c["query_identity"] for c in fifth["cohorts"]]:
        raise ValueError("current and historical query identities differ")
    folds = []
    for i, fold in enumerate(plan["folds"]):
        query = pd.DataFrame(fold["queries"])
        ids = query.row_id.to_numpy()
        ranks, raw_probabilities = read_reference(paths[i], ids)
        predictions = {REFERENCE: ranks}
        probability_diagnostics = {REFERENCE: raw_probabilities}
        y = frame.set_index("row_id").loc[ids].rule_violation.to_numpy(int)
        if not np.isclose(
            roc_auc_score(y, predictions[REFERENCE]), config["reference_auc"][i], atol=1e-10, rtol=0
        ):
            raise ValueError("reference does not reproduce its published development AUC")
        for model in VARIANTS:
            round_number = config["model_round"][model]
            report, directory = reports[round_number]
            expected = {"run_id": report["run_id"], "fold": i, "variant": model}
            p = verified_checkpoint(directory / f"fold_{i}" / model, expected)
            with np.load(p, allow_pickle=False) as data:
                ordered_ids(data["row_ids"], ids)
                predictions[model] = probabilities(data["probability"], len(ids))
                probability_diagnostics[model] = predictions[model]
            old_auc = next(
                m["auc"] for m in report["metrics"] if m["fold"] == i and m["variant"] == model
            )
            if not np.isclose(roc_auc_score(y, predictions[model]), old_auc, atol=1e-12, rtol=0):
                raise ValueError("candidate scores do not reproduce saved AUC")
        folds.append(
            {
                "rule": fold["rule"],
                "query": query,
                "y": y,
                "predictions": predictions,
                "probabilities": probability_diagnostics,
            }
        )
    return folds, cohorts


def read_reference(path, expected_ids):
    """Match historical margin ranking, not saturated float32 probability ties."""
    with np.load(path, allow_pickle=False) as data:
        ordered_ids(data["query_row_ids"], expected_ids)
        scores = data["query_adapted_scores"]
        if scores.shape != (len(expected_ids), 3) or not np.isfinite(scores).all():
            raise ValueError("reference score schema differs")
        margin = scores[:, 1].astype(float)
    ranks = (rankdata(margin, method="average") - 0.5) / len(margin)
    return ranks, expit(margin)
