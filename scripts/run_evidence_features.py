"""Round 4: training-only lexical evidence weights with cached reference fits."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

from jigsaw_rules.data import normalize, validate_frame
from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest
from scripts import run_policy_features as policy
from scripts.evidence_features import RuleEvidence, row_norms
from scripts.run_behavioral_features import (
    environment,
    metric_row,
    protocol,
    stage_read,
    stage_write,
)
from scripts.run_relational_features import fit_candidate

CONFIG = "configs/evidence_features.json"
NOTEBOOK = "notebooks/09_lexical_evidence_investigation.ipynb"
PUBLIC = "reports/evidence_features"
PRIVATE = "runs/evidence_features"
SOURCE_PATHS = (
    "scripts/evidence_features.py",
    "scripts/run_evidence_features.py",
    "tests/test_evidence_features.py",
    CONFIG,
    *policy.SOURCE_PATHS,
)
CONTROLS = (*policy.CONTROLS, "copy_lexical", "condition_lexical")
ANCHOR = "condition_lexical"
NEW = {
    "global_both": ("global", ("word", "char")),
    "rule_words": ("rule", ("word",)),
    "rule_chars": ("rule", ("char",)),
    "rule_both": ("rule", ("word", "char")),
    "permuted_both": ("permuted", ("word", "char")),
    "unshrunk_both": ("unshrunk", ("word", "char")),
}
VARIANTS = (*CONTROLS, *NEW)
CONTRASTS = [(name, ANCHOR, name + " vs policy anchor") for name in NEW]
CONTRASTS += [
    ("rule_both", "global_both", "Rule evidence versus global evidence"),
    ("rule_both", "permuted_both", "Learned weights versus permuted weights"),
    ("rule_both", "unshrunk_both", "Effect of shrinkage"),
    ("rule_both", "rule_words", "Ablation: character reweighting"),
    ("rule_both", "rule_chars", "Ablation: word reweighting"),
    (ANCHOR, "add_act_roles", "Round 3 policy gain (reused)"),
]


def hashed_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def identity(root, config):
    return {
        "source": {p: digest(root / p) for p in SOURCE_PATHS},
        "config": config,
        "environment": environment(),
        "round3_results_sha256": digest(root / policy.PUBLIC / "results.json"),
        "train_sha256": digest(root / "data/raw/train.csv"),
    }


def verify_prior(root, config):
    """Read all three prior receipts; never refit any completed reference."""
    path = root / policy.PUBLIC / "results.json"
    if digest(path) != config["round3_results_sha256"]:
        raise ValueError("round-3 results changed; preserve them")
    third = json.loads(path.read_text())
    if third["run_id"] != config["round3_run_id"]:
        raise ValueError("round-3 run identity mismatch")
    if third["identity"]["environment"] != environment():
        raise ValueError("environment changed since round 3")
    for name, sha in third["identity"]["source"].items():
        if digest(root / name) != sha:
            raise ValueError("round-3 source changed: " + name)
    first, old, second, directory2 = policy.verify_prior(root, third["identity"]["config"])
    directory3 = root / policy.PRIVATE / third["run_id"]
    marker = json.loads((directory3 / "finished.json").read_text())
    if marker["identity"] != third["identity"]:
        raise ValueError("round-3 finished identity mismatch")
    for name, sha in marker["public_hashes"].items():
        if Path(name).name != name or digest(root / policy.PUBLIC / name) != sha:
            raise ValueError("round-3 public checksum mismatch")
    for fold in range(2):
        for name in policy.NEW:
            key = {"run_id": third["run_id"], "fold": fold, "variant": name}
            if stage_read(directory3 / f"fold_{fold}" / name, key) is None:
                raise ValueError("round-3 checkpoint missing; do not refit")
    return first, old, second, directory2, third, directory3


def lexical_parts(data, spec):
    """Reconstruct existing vocabularies and prove the family partition matches."""
    train, query = data["train"], data["query"]
    tx = ("RULE: " + train.rule + "\nBODY: " + train.body).tolist()
    qx = ("RULE: " + query.rule + "\nBODY: " + query.body).tolist()
    vectorizers = {
        "word": TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=spec["word_features"],
            sublinear_tf=True,
        ),
        "char": TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=spec["char_features"],
            sublinear_tf=True,
        ),
    }
    parts, start = {}, 0
    for name, vectorizer in vectorizers.items():
        x = vectorizer.fit_transform(tx).tocsr()
        v = vectorizer.transform(qx).tocsr()
        width = x.shape[1]
        for full, chunk in ((data["base_x"], x), (data["base_v"], v)):
            difference = full[:, start : start + width] - chunk
            if np.max(np.abs(difference.data), initial=0.0) > 1e-12:
                raise ValueError("lexical family reconstruction parity failed")
        parts[name] = {
            "x": x,
            "v": v,
            "names": vectorizer.get_feature_names_out(),
        }
        start += width
    width = next(item["base_columns"] for item in data["metadata"] if item["family"] == "lexical")
    if start != width:
        raise ValueError("lexical family widths differ from the prior design")
    return parts


def prepare_fold(fold, index, prior, root, config):
    first, old, second, directory2, third, directory3 = prior
    data = policy.prepare_fold(
        fold,
        index,
        first,
        old,
        second,
        directory2,
        root,
        third["identity"]["config"],
    )
    if set(data["query"].columns) != {"row_id", "body", "rule"}:
        raise ValueError("query features must remain target-free")
    for name in ("copy_lexical", ANCHOR):
        _, design = policy.design(data, name)
        path = directory3 / f"fold_{index}" / name / "predictions.npz"
        p, _, _ = policy.load_prediction(path, data["query"].row_id.to_numpy(), design)
        data["refs"][name] = {
            "row_ids": data["query"].row_id.to_numpy(),
            "probability": p,
        }
    parts = lexical_parts(data, first["identity"]["config"])
    for index_family, part in enumerate(parts.values()):
        part["transform"] = RuleEvidence(
            alpha=config["alpha"],
            shrinkage=config["shrinkage_pairs"],
            cap=config["weight_cap"],
            seed=config["permutation_seed"] + index_family,
        ).fit(
            part["x"],
            data["y"],
            data["train"].rule,
            data["train"].repeat.to_numpy(float),
        )
    data["lexical_parts"] = parts
    data["mapper"] = policy.PolicyMap().fit(data["train"].rule)
    return data


def design(data, variant):
    mode, enabled = NEW[variant]
    chunks_x, chunks_v, checks = [], [], []
    for family, part in data["lexical_parts"].items():
        x, v = part["x"], part["v"]
        info_x = info_v = {"row_norm_max_abs_error": 0.0}
        if family in enabled:
            x, info_x = part["transform"].transform(x, data["train"].rule, mode=mode)
            v, info_v = part["transform"].transform(v, data["query"].rule, mode=mode)
        chunks_x.append(x)
        chunks_v.append(v)
        checks.append(
            {
                "family": family,
                "weighted": family in enabled,
                "training_error": info_x["row_norm_max_abs_error"],
                "query_error": info_v["row_norm_max_abs_error"],
            }
        )
    extra_x, info = data["mapper"].augment(
        sparse.hstack(chunks_x, format="csr"), data["train"].rule, mode="condition"
    )
    extra_v, _ = data["mapper"].augment(
        sparse.hstack(chunks_v, format="csr"), data["query"].rule, mode="condition"
    )
    x = sparse.hstack([data["base_x"], extra_x], format="csr")
    v = sparse.hstack([data["base_v"], extra_v], format="csr")
    old_x, old_v = policy.design(data, ANCHOR)
    for actual, baseline in ((x, old_x), (v, old_v)):
        if actual.shape != baseline.shape or actual.nnz != baseline.nnz:
            raise ValueError("candidate and anchor dimensions or sparsity differ")
        if not np.allclose(row_norms(actual), row_norms(baseline), rtol=1e-11, atol=1e-11):
            raise ValueError("candidate and anchor row norms differ")
    return x, v, {"families": checks, "added_columns": info["columns"]}


def stability(prepared):
    """Compare named training features privately; export only overlap aggregates."""
    rows = []
    for family in ("word", "char"):
        left = prepared[0]["lexical_parts"][family]
        right = prepared[1]["lexical_parts"][family]
        for rule in sorted(set(left["transform"].local_) & set(right["transform"].local_)):
            selected = []
            sizes = []
            for part in (left, right):
                weights = part["transform"].weights(rule, "rule")
                size = max(1, int(np.ceil(len(weights) * 0.10)))
                order = np.argsort(-weights, kind="stable")[:size]
                selected.append(set(part["names"][order]))
                sizes.append(len(weights))
            union = selected[0] | selected[1]
            rows.append(
                {
                    "family": family,
                    "policy": rule,
                    "highest_weight_decile_jaccard": len(selected[0] & selected[1]) / len(union),
                    "fold0_vocabulary": sizes[0],
                    "fold1_vocabulary": sizes[1],
                    "interpretation": "descriptive weight stability, not predictive proof",
                }
            )
    return rows


def comparisons(records, frame, config):
    from scipy.stats import rankdata
    from sklearn.metrics import roc_auc_score

    from scripts.run_behavioral_features import weighted_auc_samples

    ids = sorted({int(i) for rec in records.values() for i in rec["row_ids"]})
    lookup = frame.set_index("row_id")
    codes, labels = pd.factorize(lookup.loc[ids].body.map(normalize), sort=True)
    group_map = dict(zip(ids, codes, strict=True))
    rng = np.random.default_rng(config["bootstrap_seed"])
    reps = config["bootstrap_replicates"]
    weights = rng.multinomial(len(labels), np.full(len(labels), 1 / len(labels)), size=reps)
    points, samples, pooled = {}, {}, []
    for name in VARIANTS:
        scores, draws, truths, probs, ranks = [], [], [], [], []
        for fold in range(2):
            rec = records[(fold, name)]
            y = lookup.loc[rec["row_ids"]].rule_violation.to_numpy(dtype=int)
            p = rec["probability"]
            scores.append(float(roc_auc_score(y, p)))
            w = weights[:, [group_map[int(i)] for i in rec["row_ids"]]]
            draws.append(weighted_auc_samples(y, p, w))
            truths.append(y)
            probs.append(p)
            ranks.append((rankdata(p, method="average") - 0.5) / len(p))
        points[name] = float(np.mean(scores))
        samples[name] = np.mean(draws, axis=0)
        pooled.append(
            {
                "variant": name,
                "macro_auc": points[name],
                "pooled_auc": float(roc_auc_score(np.concatenate(truths), np.concatenate(probs))),
                "ranked_pooled_auc": float(
                    roc_auc_score(np.concatenate(truths), np.concatenate(ranks))
                ),
            }
        )
    delta = np.asarray([points[a] - points[b] for a, b, _ in CONTRASTS])
    boot = np.column_stack([samples[a] - samples[b] for a, b, _ in CONTRASTS])
    valid = np.isfinite(boot).all(axis=1)
    if valid.sum() < 0.9 * reps:
        raise ValueError("insufficient valid paired bootstrap draws")
    band = float(np.quantile(np.max(np.abs(boot[valid] - delta), axis=1), 0.95))
    out = []
    for (a, b, title), d in zip(CONTRASTS, delta, strict=True):
        out.append(
            {
                "comparison": title,
                "candidate": a,
                "reference": b,
                "delta_auc": float(d),
                "simultaneous_low": float(d - band),
                "simultaneous_high": float(d + band),
                "valid_draws": int(valid.sum()),
            }
        )
    return out, pooled


def decide(metrics, contrasts, pooled, config):
    requirements = []
    primary = config["primary"]
    for ref in (ANCHOR, "permuted_both"):
        contrast = next(c for c in contrasts if c["candidate"] == primary and c["reference"] == ref)
        deltas = []
        for fold in range(2):
            m = {row["variant"]: row["auc"] for row in metrics if row["fold"] == fold}
            deltas.append(float(m[primary] - m[ref]))
        passed = (
            contrast["delta_auc"] >= config["minimum_macro_delta"]
            and contrast["simultaneous_low"] > 0
            and min(deltas) >= 0
        )
        requirements.append(
            {
                "reference": ref,
                "contrast": contrast,
                "per_policy_delta": deltas,
                "passed": bool(passed),
            }
        )
    ranks = {p["variant"]: p["ranked_pooled_auc"] for p in pooled}
    delta = ranks[primary] - ranks[ANCHOR]
    passed = all(r["passed"] for r in requirements) and delta >= 0
    return {
        "decision": "ELIGIBLE_FOR_NEXT_VALIDATION_ONLY" if passed else "DO_NOT_PROMOTE_PRIMARY",
        "primary_requirements": requirements,
        "primary_ranked_pooled_delta": float(delta),
    }


def run_study(root: Path, *, max_new_fits=None):
    started = time.monotonic()
    config = json.loads((root / CONFIG).read_text())
    fixed = {
        "primary": "rule_both",
        "new_fits": 12,
        "cached_control_fits": 10,
        "alpha": 1.0,
        "shrinkage_pairs": 64.0,
        "weight_cap": 3.0,
        "permutation_seed": 20260911,
        "automatic_gpu_authorization": False,
    }
    if any(config.get(k) != v for k, v in fixed.items()):
        raise ValueError("registered design differs; preserve the existing experiment")
    if not 0 < config["max_seconds"] <= 240:
        raise ValueError("invalid scientific runtime cap")
    if digest(root / "data/raw/train.csv") != config["train_sha256"]:
        raise ValueError("raw training checksum mismatch")
    prior = verify_prior(root, config)
    frame = pd.read_csv(root / "data/raw/train.csv")
    validate_frame(frame, train=True)
    ident = identity(root, config)
    run_id = hashed_json(ident)[:20]
    work, public = root / PRIVATE / run_id, root / PUBLIC
    work.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    with (
        FileLock(str(root / PRIVATE / "study.lock"), timeout=1),
        threadpool_limits(limits=1),
        Progress(work / "events.jsonl", "evidence_round4", heartbeat_seconds=15) as log,
    ):
        done = work / "finished.json"
        if done.exists():
            complete = json.loads(done.read_text())
            if complete["identity"] != ident:
                raise ValueError("round-4 completion identity differs")
            for name, sha in complete["public_hashes"].items():
                if Path(name).name != name or digest(public / name) != sha:
                    raise ValueError("round-4 public checksum mismatch")
            for fold in range(2):
                for name in NEW:
                    key = {"run_id": run_id, "fold": fold, "variant": name}
                    if stage_read(work / f"fold_{fold}" / name, key) is None:
                        raise ValueError("completed round-4 checkpoint missing")
            atomic_json(work / "last_invocation.json", {"new_fits": 0, "reused_new_fits": 12})
            log.emit("completed_run_reused", new_fits=0, reused_new_fits=12)
            return json.loads((public / "results.json").read_text())
        # Avoid replacing a different completed public study with changed source/config.
        if (public / "results.json").exists():
            previous = json.loads((public / "results.json").read_text())
            if previous.get("run_id") != run_id:
                raise ValueError("different completed round-4 result exists; preserve it")
        plan, cohorts = protocol(frame, config["query_counts"])
        expected = [c["query_identity"] for c in prior[4]["cohorts"]]
        if [c["query_identity"] for c in cohorts] != expected:
            raise ValueError("historical query identity changed")
        atomic_json(work / "identity.json", ident)
        prepared = []
        for index, fold in enumerate(plan["folds"]):
            prepared.append(prepare_fold(fold, index, prior, root, config))
        log.emit("all_control_designs_verified", cached_controls=10, new_fits=0)
        records, metrics, profiles, checks, sizes = {}, [], [], [], []
        new_fits = reused = 0
        for index, data in enumerate(prepared):
            for name, part in data["lexical_parts"].items():
                profiles.extend(part["transform"].profile(family=name, fold=index))
            for name, rec in data["refs"].items():
                records[(index, name)] = rec
            for name in NEW:
                if time.monotonic() - started > config["max_seconds"]:
                    raise TimeoutError("CPU budget exhausted; valid stages preserved")
                folder = work / f"fold_{index}" / name
                key = {"run_id": run_id, "fold": index, "variant": name}
                x, v, info = design(data, name)
                checks.append({"fold": index, "variant": name, **info})
                sizes.append(
                    {
                        "fold": index,
                        "variant": name,
                        "columns": x.shape[1],
                        "nonzeros": x.nnz,
                        "training_rows": x.shape[0],
                    }
                )
                if stage_read(folder, key) is None:
                    if max_new_fits is not None and new_fits >= max_new_fits:
                        raise TimeoutError("authored interruption for recovery test")
                    p, coef, intercept = fit_candidate(
                        x,
                        data["y"],
                        v,
                        data["train"].repeat.to_numpy(float),
                        prior[0]["identity"]["config"],
                    )
                    out = io.BytesIO()
                    np.savez_compressed(
                        out,
                        row_ids=data["query"].row_id.to_numpy(),
                        probability=p,
                        coefficients=coef,
                        intercept=intercept,
                    )
                    stage_write(folder, {"predictions.npz": out.getvalue()}, key)
                    new_fits += 1
                else:
                    reused += 1
                p, _, _ = policy.load_prediction(
                    folder / "predictions.npz", data["query"].row_id.to_numpy(), v
                )
                records[(index, name)] = {
                    "row_ids": data["query"].row_id.to_numpy(),
                    "probability": p,
                }
                log.emit(
                    "candidate_complete",
                    completed=new_fits + reused,
                    total=12,
                    fold=index,
                    variant=name,
                    new_fits=new_fits,
                    reused_fits=reused,
                )
        lookup = frame.set_index("row_id")
        for (index, name), rec in records.items():
            labels = lookup.loc[rec["row_ids"]].rule_violation.to_numpy(dtype=int)
            metrics.append(
                metric_row(index, name, rec["probability"], labels, plan["folds"][index]["rule"])
            )
        contrasts, pooled = comparisons(records, frame, config)
        result = {
            "schema": 1,
            "status": "EVIDENCE_ROUND_COMPLETE",
            "run_id": run_id,
            "identity": ident,
            "round3_run_id": prior[4]["run_id"],
            "new_fits": new_fits,
            "reused_new_fits": reused,
            "reused_prior_controls": 10,
            "candidate_fits": 12,
            "cohorts": cohorts,
            "control_design_parity": True,
            "metrics": metrics,
            "pooled_metrics": pooled,
            "comparisons": contrasts,
            "weight_profiles": profiles,
            "norm_checks": checks,
            "weight_stability": stability(prepared),
            "design_sizes": sizes,
            **decide(metrics, contrasts, pooled, config),
            "new_neural_inference": 0,
            "gpu": False,
            "automatic_gpu_authorization": False,
            "kaggle_score": None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "limitations": [
                "Adaptive exploratory follow-up; not an independent holdout.",
                "The rule-conditioned anchor was not promoted.",
                "Training-only label evidence; query targets enter scoring only.",
                "Supplied labels exist for the query rule: not zero-shot.",
                "The permutation control fixes weight distributions, not all capacity.",
                "Equal row norms do not equalize per-coordinate regularization.",
                "Intervals cover 12 comparisons, not the full adaptive history.",
                "No comparison to accepted-model predictions or new Kaggle score.",
                "One permutation and one shrinkage strength; no tuning sweep.",
                "Historical generic NB weighting is not evidence for this design.",
            ],
        }
        atomic_json(public / "results.json", result)
        atomic_json(
            public / "feature_catalog.json",
            {
                "feature_family": "rule-conditioned lexical evidence weights",
                "vocabulary": "unchanged training-only word and character features",
                "new_dimensions": 0,
                "changes": "relative feature weights in the policy-specific block",
                "unchanged": "shared lexical block and screened dense columns",
                "counts": "training-pair document incidence, not token frequency",
                "normalization": "each word and character row norm preserved",
                "profiles": profiles,
            },
        )
        atomic_json(
            done,
            {
                "identity": ident,
                "public_hashes": {
                    n: digest(public / n) for n in ("results.json", "feature_catalog.json")
                },
            },
        )
        atomic_json(
            work / "last_invocation.json",
            {"new_fits": new_fits, "reused_new_fits": reused},
        )
        log.emit("results_saved", decision=result["decision"], new_fits=new_fits)
        return result


def figures(result):
    import plotly.graph_objects as go

    metrics = pd.DataFrame(result["metrics"])
    charts = []
    fig = go.Figure()
    for rule, rows in metrics.groupby("policy", sort=True):
        label = "Advertising" if "advert" in rule.lower() else "Legal advice"
        fig.add_bar(x=rows.variant, y=rows.auc, name=label)
    fig.update_layout(
        title="01 | Policy AUC: cached anchors and evidence weighting",
        barmode="group",
    )
    charts.append(fig)
    rows = result["comparisons"]
    fig = go.Figure(
        go.Scatter(
            x=[r["delta_auc"] for r in rows],
            y=[r["comparison"] for r in rows],
            mode="markers",
            error_x={"array": [r["simultaneous_high"] - r["delta_auc"] for r in rows]},
        )
    )
    fig.add_vline(x=0)
    fig.update_layout(title="02 | Paired uncertainty across 12 planned contrasts", height=700)
    charts.append(fig)
    chosen = [
        r
        for r in rows
        if r["candidate"] == "rule_both"
        and r["reference"] in ("global_both", "permuted_both", "unshrunk_both")
    ]
    fig = go.Figure(go.Bar(x=[r["comparison"] for r in chosen], y=[r["delta_auc"] for r in chosen]))
    fig.add_hline(y=0)
    fig.update_layout(title="03 | Which part of the weighting has measured value?")
    charts.append(fig)
    chosen = [r for r in rows if r["comparison"].startswith("Ablation:")]
    fig = go.Figure(go.Bar(x=[r["comparison"] for r in chosen], y=[r["delta_auc"] for r in chosen]))
    fig.add_hline(y=0)
    fig.update_layout(title="04 | Incremental word and character-family value")
    charts.append(fig)
    profiles = [r for r in result["weight_profiles"] if r["mode"] == "rule"]
    labels = [
        f"F{r['fold']} | {r['family']} | " + ("ads" if "advert" in r["policy"] else "advice")
        for r in profiles
    ]
    fig = go.Figure(
        go.Scatter(
            x=labels,
            y=[r["median"] for r in profiles],
            mode="markers",
            error_y={
                "type": "data",
                "symmetric": False,
                "array": [r["q75"] - r["median"] for r in profiles],
                "arrayminus": [r["median"] - r["q25"] for r in profiles],
            },
        )
    )
    fig.update_layout(title="05 | Training-fitted weights: median and interquartile range")
    charts.append(fig)
    profiles = [r for r in profiles if r["family"] == "word"]
    fig = go.Figure()
    for r in profiles:
        label = f"F{r['fold']} | " + ("ads" if "advert" in r["policy"] else "advice")
        fig.add_scatter(
            x=[min(r["permitted_pairs"], r["violating_pairs"])],
            y=[r["local_evidence_fraction"]],
            mode="markers",
            name=label,
            marker={"size": 12},
        )
    fig.update_layout(
        title="06 | Less rule evidence means more global backoff",
        xaxis_title="Unique pairs in the smaller training class",
        yaxis_title="Local evidence fraction",
    )
    charts.append(fig)
    rows = result["weight_stability"]
    fig = go.Figure(
        go.Bar(
            x=[
                r["family"] + " | " + ("ads" if "advert" in r["policy"] else "advice") for r in rows
            ],
            y=[r["highest_weight_decile_jaccard"] for r in rows],
        )
    )
    fig.update_layout(title="07 | Highest-weight feature overlap across folds", yaxis_range=[0, 1])
    charts.append(fig)
    fig = go.Figure()
    for field in ("brier", "log_loss"):
        values = metrics.groupby("variant")[field].mean()
        fig.add_bar(x=values.index, y=values.values, name=field)
    fig.update_layout(title="08 | Probability diagnostics: lower is better", barmode="group")
    charts.append(fig)
    for fig in charts:
        fig.update_layout(
            template="plotly_white",
            height=fig.layout.height or 540,
            margin={"l": 125, "r": 35, "t": 90, "b": 160},
            font={"size": 13},
        )
        fig.update_xaxes(automargin=True)
        fig.update_yaxes(automargin=True)
    return charts


def bounded_compute(root: Path):
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "scripts.run_evidence_features",
            "--compute",
            "--root",
            str(root),
        ],
        cwd=root,
        start_new_session=True,
    )
    try:
        code = proc.wait(timeout=300)
    except BaseException:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        raise
    if code:
        raise RuntimeError("evidence worker stopped; preserve checkpoints and return ZIP")
    return json.loads((root / PUBLIC / "results.json").read_text())


def write_dashboard(root: Path, result):
    import html

    import plotly.io as pio

    title = "Jigsaw | Round 4: rule-specific lexical evidence"
    parts = [
        "<!doctype html><html><meta charset='utf-8'><title>" + title + "</title>",
        "<body><h1>" + title + ("</h1><p>Exploratory CPU feature study. Not a Kaggle score.</p>"),
        "<p>Decision: " + html.escape(result["decision"]) + "</p>",
    ]
    for i, fig in enumerate(figures(result)):
        parts.append(pio.to_html(fig, full_html=False, include_plotlyjs=(i == 0)))
    parts.append("</body></html>")
    path = root / PUBLIC / "dashboard.html"
    atomic_bytes(path, "\n".join(parts).encode())
    return path


def notebook_source_hash(path: Path):
    import nbformat

    n = nbformat.read(path, as_version=4)
    return hashed_json([(c.cell_type, c.source) for c in n.cells])


def execute_notebook(root: Path):
    import nbformat
    from nbclient import NotebookClient

    target = root / NOTEBOOK
    work = root / PRIVATE
    work.mkdir(parents=True, exist_ok=True)
    config = json.loads((root / CONFIG).read_text())
    marker = work / "notebook_execution.json"
    ident = {
        "scientific_identity": identity(root, config),
        "source_hash": notebook_source_hash(target),
    }
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["identity"] != ident or digest(target) != saved["notebook_sha256"]:
            raise ValueError("notebook identity changed; saved work preserved")
        for name, sha in saved["artifact_hashes"].items():
            if digest(root / PUBLIC / name) != sha:
                raise ValueError("notebook output checksum mismatch")
        return {**saved, "status": "NOTEBOOK_REUSED", "cells_reexecuted": 0}
    n = nbformat.read(target, as_version=4)
    for cell in n.cells:
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
    try:
        NotebookClient(
            n, timeout=330, kernel_name="jigsaw-rules", resources={"metadata": {"path": str(root)}}
        ).execute()
    except BaseException:
        nbformat.write(n, work / "interrupted_notebook.ipynb")
        raise
    cells = [c for c in n.cells if c.cell_type == "code"]
    count = sum(
        "application/vnd.plotly.v1+json" in o.get("data", {}) for c in cells for o in c.outputs
    )
    if any(c.execution_count is None for c in cells) or count != 8:
        raise ValueError("notebook incomplete or Plotly output count differs")
    atomic_bytes(target, nbformat.writes(n).encode())
    record = {
        "status": "NOTEBOOK_EXECUTED",
        "identity": ident,
        "notebook_sha256": digest(target),
        "code_cells": len(cells),
        "plotly_charts": count,
        "artifact_hashes": {
            name: digest(root / PUBLIC / name) for name in ("results.json", "dashboard.html")
        },
    }
    atomic_json(marker, record)
    return record


def export_return(root: Path, destination: Path | None = None):
    destination = destination or Path.home() / "jigsaw_feature_round4_return.zip"
    names = [
        "scripts/evidence_features.py",
        "scripts/run_evidence_features.py",
        CONFIG,
        "tests/test_evidence_features.py",
        "docs/EVIDENCE_FEATURES.md",
        NOTEBOOK,
        PUBLIC + "/results.json",
        PUBLIC + "/feature_catalog.json",
        PRIVATE + "/notebook_execution.json",
        PRIVATE + "/replay.json",
        PRIVATE + "/tests.xml",
        PRIVATE + "/install.json",
        PRIVATE + "/launcher_report.json",
        "reports/behavioral_features/results.json",
        "reports/relational_features/results.json",
        "reports/policy_features/results.json",
    ]
    payload = {n: (root / n).read_bytes() for n in names if (root / n).is_file()}
    payload["SHA256SUMS.json"] = json.dumps(
        {n: hashlib.sha256(b).hexdigest() for n, b in payload.items()}, indent=2, sort_keys=True
    ).encode()
    temp = destination.with_suffix(".zip.partial")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in payload.items():
            z.writestr(name, data)
    os.replace(temp, destination)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--compute", action="store_true")
    parser.add_argument("--execute-notebook", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.export_only:
        print("RETURN_FILE:", export_return(root))
        return
    if args.execute_notebook:
        first = execute_notebook(root)
        second = execute_notebook(root)
        atomic_json(
            root / PRIVATE / "replay.json",
            {
                "first": first["status"],
                "second": second["status"],
                "cells_reexecuted": second["cells_reexecuted"],
            },
        )
        result = bounded_compute(root)
        print("RESULT:", result["status"])
        print("DECISION:", result["decision"])
        return
    if not args.compute:
        parser.error("select --compute, --execute-notebook or --export-only")
    budget = json.loads((root / CONFIG).read_text())["max_seconds"]
    if not isinstance(budget, int) or not 0 < budget <= 240 or not hasattr(signal, "setitimer"):
        raise ValueError("POSIX and positive runtime budget <=240 seconds required")

    def expired(signum, frame):
        raise TimeoutError("hard runtime limit; successful checkpoints preserved")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, budget)
    try:
        result = run_study(root)
        print("RESULT:", result["status"], "DECISION:", result["decision"])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    main()
