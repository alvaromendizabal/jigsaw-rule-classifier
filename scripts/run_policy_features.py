"""Round 3: policy interactions with matched controls and reusable CPU checkpoints."""

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
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from jigsaw_rules.data import normalize, validate_frame
from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest
from scripts.behavioral_features import basic_features
from scripts.policy_features import PolicyMap, matched_blocks
from scripts.relational_features import CATALOG, relational_features
from scripts.run_behavioral_features import (
    environment,
    metric_row,
    protocol,
    stage_read,
    stage_write,
    weighted_auc_samples,
)
from scripts.run_relational_features import SOURCE_PATHS as ROUND2_SOURCES
from scripts.run_relational_features import fit_candidate, lexical_matrices, transform_block
from scripts.run_relational_features import verify_prior as verify_round1

CONFIG = "configs/policy_features.json"
NOTEBOOK = "notebooks/08_policy_feature_investigation.ipynb"
PUBLIC = "reports/policy_features"
PRIVATE = "runs/policy_features"
SOURCE_PATHS = (
    "scripts/policy_features.py",
    "scripts/run_policy_features.py",
    "tests/test_policy_features.py",
    CONFIG,
    *ROUND2_SOURCES,
)
CONTROLS = ("lexical_control", "add_behavior", "add_act_roles")
ANCHOR = "add_act_roles"
NEW = {
    "copy_lexical": ("copy", "lexical"),
    "condition_lexical": ("condition", "lexical"),
    "copy_behavior": ("copy", "behavior"),
    "condition_behavior": ("condition", "behavior"),
    "copy_both": ("copy", "both"),
    "condition_both": ("condition", "both"),
}
VARIANTS = (*CONTROLS, *NEW)
CONTRASTS = [(name, ANCHOR, name + " vs actor anchor") for name in NEW]
CONTRASTS += [
    ("condition_" + f, "copy_" + f, "conditioning: " + f) for f in ("lexical", "behavior", "both")
]
CONTRASTS += [
    ("condition_both", "condition_behavior", "removal: policy lexical"),
    ("condition_both", "condition_lexical", "removal: policy behavior"),
    (ANCHOR, "add_behavior", "Round 2 actor gain (reused)"),
    ("add_behavior", "lexical_control", "Round 1 behavior gain (reused)"),
]


def hashed_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def identity(root, config):
    return {
        "source": {p: digest(root / p) for p in SOURCE_PATHS},
        "config": config,
        "environment": environment(),
        "round2_results_sha256": digest(root / "reports/relational_features/results.json"),
        "train_sha256": digest(root / "data/raw/train.csv"),
    }


def verify_prior(root, config):
    path = root / "reports/relational_features/results.json"
    if digest(path) != config["round2_results_sha256"]:
        raise ValueError("round-2 results changed; preserve them")
    previous = json.loads(path.read_text())
    if previous["run_id"] != config["round2_run_id"]:
        raise ValueError("round-2 run identity mismatch")
    if previous["identity"]["environment"] != environment():
        raise ValueError("environment changed since round 2")
    for name, sha in previous["identity"]["source"].items():
        if digest(root / name) != sha:
            raise ValueError("round-2 source changed: " + name)
    prior, old = verify_round1(root, previous["identity"]["config"])
    directory = root / "runs/relational_features" / previous["run_id"]
    marker = json.loads((directory / "finished.json").read_text())
    if marker["identity"] != previous["identity"]:
        raise ValueError("round-2 finished identity mismatch")
    for name, sha in marker["public_hashes"].items():
        if Path(name).name != name or digest(root / "reports/relational_features" / name) != sha:
            raise ValueError("round-2 public checksum mismatch")
    variants = sorted({m["variant"] for m in previous["metrics"]} - set(CONTROLS[:2]))
    for fold in range(2):
        for name in variants:
            rec = stage_read(
                directory / f"fold_{fold}" / name,
                {"run_id": previous["run_id"], "fold": fold, "variant": name},
            )
            if rec is None:
                raise ValueError("round-2 checkpoint missing; do not refit old work")
    return prior, old, previous, directory


def load_prediction(path, ids, design=None):
    with np.load(path, allow_pickle=False) as data:
        if not np.array_equal(data["row_ids"], ids):
            raise ValueError("cached prediction row order mismatch")
        p = data["probability"].copy()
        coef = data["coefficients"].ravel().copy()
        intercept = data["intercept"].ravel().copy()
    if p.shape != ids.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("invalid cached probabilities")
    if not np.isfinite(coef).all() or intercept.size != 1 or not np.isfinite(intercept).all():
        raise ValueError("invalid cached coefficients")
    if design is not None:
        if design.shape[1] != len(coef):
            raise ValueError("cached design dimension mismatch")
        actual = expit(np.asarray(design @ coef).ravel() + intercept[0])
        if not np.allclose(actual, p, atol=1e-10, rtol=1e-10):
            raise ValueError("cached control design parity failed; no new fit allowed")
    return p, coef, intercept


def prepare_fold(fold, fold_idx, prior, old, previous, directory, root, config):
    train = pd.DataFrame(fold["training"])
    query = pd.DataFrame(fold["queries"])
    if set(query.columns) != {"body", "rule", "row_id"}:
        raise ValueError("query feature frame must be target-free")
    if set(train.body.map(normalize)) & set(query.body.map(normalize)):
        raise ValueError("query text leakage in training")
    y = train.rule_violation.to_numpy(dtype=int)
    spec = prior["identity"]["config"]
    lx, lv = lexical_matrices(train, query, spec)
    basics = [basic_features(t[["body", "rule"]]) for t in (train, query)]
    behavior_names = [n for n in basics[0] if n.startswith("behavior/")]
    bx, bv, bmeta = transform_block(*basics, y, behavior_names, spec["feature_budget_per_family"])
    rx = relational_features(train[["body"]])
    rv = relational_features(query[["body"]])
    ax, av, ameta = transform_block(
        rx, rv, y, CATALOG["act_roles"], previous["identity"]["config"]["feature_budget_per_family"]
    )
    old_manifest = json.loads((old / f"fold_{fold_idx}/feature_manifest.json").read_text())
    if bmeta["names"] != old_manifest["names"]["behavior"]:
        raise ValueError("behavior selection changed")
    expected = next(
        s for s in previous["selection"] if s["fold"] == fold_idx and s["family"] == "act_roles"
    )
    for key in ("names", "scaler_mean", "scaler_scale"):
        if ameta[key] != expected[key]:
            raise ValueError("actor-role selection/scaling changed")
    dense_x = sparse.csr_matrix(np.hstack([bx, ax]))
    dense_v = sparse.csr_matrix(np.hstack([bv, av]))
    base_x = sparse.hstack([lx, dense_x], format="csr")
    base_v = sparse.hstack([lv, dense_v], format="csr")
    refs = {}
    for name, folder, design in (
        ("lexical_control", old, lv),
        ("add_behavior", old, sparse.hstack([lv, bv], format="csr")),
        (ANCHOR, directory, base_v),
    ):
        p, _, _ = load_prediction(
            folder / f"fold_{fold_idx}" / name / "predictions.npz", query.row_id.to_numpy(), design
        )
        refs[name] = {"row_ids": query.row_id.to_numpy(), "probability": p}
    mapper = PolicyMap().fit(train.rule)
    blocks = {}
    metadata = []
    for family, x, v in (("lexical", lx, lv), ("behavior", dense_x, dense_v)):
        px, tx = matched_blocks(x, train.rule, mapper)
        pv, tv = matched_blocks(v, query.rule, mapper)
        if tv["condition"]["unknown_rows"]:
            raise ValueError("target policy absent from eligible training supports")
        blocks[family] = (px, pv)
        metadata.append(
            {
                "fold": fold_idx,
                "family": family,
                "base_columns": x.shape[1],
                "training": tx,
                "query": tv,
                "rule_count": len(mapper.rules_),
                "row_norm_parity": True,
            }
        )
    support = []
    for name in mapper.rules_:
        rows = train[train.rule.map(normalize) == name]
        support.append(
            {
                "fold": fold_idx,
                "policy": name,
                "rows": len(rows),
                "permitted": int((rows.rule_violation == 0).sum()),
                "violating": int((rows.rule_violation == 1).sum()),
                "query_policy": normalize(fold["rule"]) == name,
            }
        )
    return {
        "train": train,
        "query": query,
        "y": y,
        "base_x": base_x,
        "base_v": base_v,
        "blocks": blocks,
        "refs": refs,
        "metadata": metadata,
        "support": support,
        "rules": mapper.rules_,
        "dense_names": bmeta["names"] + ameta["names"],
    }


def design(prepared, variant):
    mode, family = NEW[variant]
    families = ("lexical", "behavior") if family == "both" else (family,)
    x = sparse.hstack(
        [prepared["base_x"]] + [prepared["blocks"][f][0][mode] for f in families], format="csr"
    )
    v = sparse.hstack(
        [prepared["base_v"]] + [prepared["blocks"][f][1][mode] for f in families], format="csr"
    )
    return x, v


def comparisons(records, frame, config):
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
    return [
        {
            "comparison": title,
            "candidate": a,
            "reference": b,
            "delta_auc": float(d),
            "simultaneous_low": float(d - band),
            "simultaneous_high": float(d + band),
            "valid_draws": int(valid.sum()),
        }
        for (a, b, title), d in zip(CONTRASTS, delta, strict=True)
    ], pooled


def decide(metrics, contrasts, pooled, config):
    primary = config["primary"]
    requirements = []
    for ref in (ANCHOR, "copy_lexical"):
        contrast = next(c for c in contrasts if c["candidate"] == primary and c["reference"] == ref)
        per_policy = []
        for fold in range(2):
            scores = {m["variant"]: m["auc"] for m in metrics if m["fold"] == fold}
            per_policy.append(float(scores[primary] - scores[ref]))
        passed = (
            contrast["delta_auc"] >= config["minimum_macro_delta"]
            and contrast["simultaneous_low"] > 0
            and min(per_policy) >= 0
        )
        requirements.append(
            {
                "reference": ref,
                "contrast": contrast,
                "per_policy_delta": per_policy,
                "passed": bool(passed),
            }
        )
    rank = {p["variant"]: p["ranked_pooled_auc"] for p in pooled}
    rank_delta = rank[primary] - rank[ANCHOR]
    passed = all(r["passed"] for r in requirements) and rank_delta >= 0
    return {
        "decision": "ELIGIBLE_FOR_NEXT_VALIDATION_ONLY" if passed else "DO_NOT_PROMOTE_PRIMARY",
        "primary_requirements": requirements,
        "primary_ranked_pooled_delta": rank_delta,
    }


def run_study(root: Path, *, max_new_fits=None):
    started = time.monotonic()
    config = json.loads((root / CONFIG).read_text())
    if (
        config["primary"] != "condition_lexical"
        or config["new_fits"] != 12
        or config["augmentation_scale"] != 1.0
        or config["automatic_gpu_authorization"]
    ):
        raise ValueError("registered design differs; do not silently change this experiment")
    if not 0 < config["max_seconds"] <= 240:
        raise ValueError("invalid scientific runtime cap")
    if digest(root / "data/raw/train.csv") != config["train_sha256"]:
        raise ValueError("raw training checksum mismatch")
    prior, old, previous, directory = verify_prior(root, config)
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
        Progress(work / "events.jsonl", "policy_round3", heartbeat_seconds=15) as log,
    ):
        done = work / "finished.json"
        if done.exists():
            complete = json.loads(done.read_text())
            if complete["identity"] != ident:
                raise ValueError("round-3 completion identity differs")
            for name, sha in complete["public_hashes"].items():
                if Path(name).name != name or digest(public / name) != sha:
                    raise ValueError("round-3 public checksum mismatch")
            for fold in range(2):
                for name in NEW:
                    if (
                        stage_read(
                            work / f"fold_{fold}" / name,
                            {"run_id": run_id, "fold": fold, "variant": name},
                        )
                        is None
                    ):
                        raise ValueError("completed round-3 checkpoint missing")
            atomic_json(work / "last_invocation.json", {"new_fits": 0, "reused_new_fits": 12})
            log.emit("completed_run_reused", new_fits=0, reused_new_fits=12)
            return json.loads((public / "results.json").read_text())
        plan, cohorts = protocol(frame, config["query_counts"])
        actual_ids = [c["query_identity"] for c in cohorts]
        expected_ids = [c["query_identity"] for c in previous["cohorts"]]
        if actual_ids != expected_ids:
            raise ValueError("historical query identity changed")
        atomic_json(work / "identity.json", ident)
        prepared = []
        # Verify all six cached reference predictions before allowing even the first new fit.
        for i, fold in enumerate(plan["folds"]):
            prepared.append(prepare_fold(fold, i, prior, old, previous, directory, root, config))
        log.emit("all_control_designs_verified", cached_controls=6, new_fits=0)
        records, metrics, blocks, support, effects, widths = {}, [], [], [], [], []
        new_fits = reused = 0
        for i, data in enumerate(prepared):
            blocks.extend(data["metadata"])
            support.extend(data["support"])
            for name, rec in data["refs"].items():
                records[(i, name)] = rec
            for name in NEW:
                if time.monotonic() - started > config["max_seconds"]:
                    raise TimeoutError("CPU budget exhausted; valid stages preserved")
                folder = work / f"fold_{i}" / name
                key = {"run_id": run_id, "fold": i, "variant": name}
                x, v = design(data, name)
                widths.append(
                    {
                        "fold": i,
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
                        prior["identity"]["config"],
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
                p, coef, _ = load_prediction(
                    folder / "predictions.npz", data["query"].row_id.to_numpy(), v
                )
                records[(i, name)] = {"row_ids": data["query"].row_id.to_numpy(), "probability": p}
                if name == "condition_behavior":
                    start = data["base_x"].shape[1]
                    d = len(data["dense_names"])
                    for j, rule in enumerate(data["rules"]):
                        for k, feature in enumerate(data["dense_names"]):
                            effects.append(
                                {
                                    "fold": i,
                                    "policy": rule,
                                    "feature": feature,
                                    "deviation_coefficient": float(coef[start + j * d + k]),
                                }
                            )
                log.emit(
                    "candidate_complete",
                    completed=new_fits + reused,
                    total=12,
                    fold=i,
                    variant=name,
                    new_fits=new_fits,
                    reused_fits=reused,
                )
        lookup = frame.set_index("row_id")
        for (i, name), rec in records.items():
            labels = lookup.loc[rec["row_ids"]].rule_violation.to_numpy(dtype=int)
            metrics.append(
                metric_row(i, name, rec["probability"], labels, plan["folds"][i]["rule"])
            )
        contrasts, pooled = comparisons(records, frame, config)
        result = {
            "schema": 1,
            "status": "POLICY_ROUND_COMPLETE",
            "run_id": run_id,
            "identity": ident,
            "round2_run_id": previous["run_id"],
            "new_fits": new_fits,
            "reused_new_fits": reused,
            "reused_prior_controls": 6,
            "candidate_fits": 12,
            "cohorts": cohorts,
            "control_design_parity": True,
            "metrics": metrics,
            "pooled_metrics": pooled,
            "comparisons": contrasts,
            "blocks": blocks,
            "support_coverage": support,
            "dense_rule_effects": effects,
            "design_sizes": widths,
            **decide(metrics, contrasts, pooled, config),
            "new_neural_inference": 0,
            "gpu": False,
            "automatic_gpu_authorization": False,
            "kaggle_score": None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "limitations": [
                "Exploratory follow-up chosen after Round 2, not a fresh holdout.",
                "Actor anchor was not promoted; its reuse is diagnostic only.",
                "The query rule has legal supplied labels in training: not zero-shot transfer.",
                "Unknown policies receive no learned extra rule block; the shared features remain.",
                "Intervals cover this round's 13 contrasts, not all adaptive project choices.",
                "No new Kaggle score or accepted-Qwen-model comparison occurs in this round.",
                "Coefficient plots show fitted associations, not causal feature value.",
            ],
        }
        atomic_json(public / "results.json", result)
        atomic_json(
            public / "feature_catalog.json",
            {
                "lexical": (
                    "training-fitted word and character columns multiplied "
                    "by normalized rule indicators"
                ),
                "behavior": (
                    "Round 1 behavior + Round 2 actor-role "
                    # Preserve this split for the formatter and E501.
                    "training-selected columns by rule"
                ),
                "controls": "same width and per-row norm; one shared copy block plus zero padding",
                "sparse_screen": (
                    "existing train-only min_df/vocabulary budget, "
                    # Preserve this split for the formatter and E501.
                    "no query-led screening"
                ),
                "dimensions": blocks,
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
            work / "last_invocation.json", {"new_fits": new_fits, "reused_new_fits": reused}
        )
        log.emit("results_saved", decision=result["decision"], new_fits=new_fits)
        return result


def figures(result):
    import plotly.graph_objects as go

    metrics = pd.DataFrame(result["metrics"])
    charts = []
    fig = go.Figure()
    for policy, rows in metrics.groupby("policy", sort=True):
        fig.add_bar(
            x=rows.variant,
            y=rows.auc,
            name="Advertising" if "advert" in policy.lower() else "Legal advice",
        )
    fig.update_layout(
        title="01 | AUC by policy: cached anchors and matched feature tests", barmode="group"
    )
    charts.append(fig)
    comparisons = result["comparisons"]
    fig = go.Figure(
        go.Scatter(
            x=[r["delta_auc"] for r in comparisons],
            y=[r["comparison"] for r in comparisons],
            mode="markers",
            error_x={"array": [r["simultaneous_high"] - r["delta_auc"] for r in comparisons]},
        )
    )
    fig.add_vline(x=0)
    fig.update_layout(
        title="02 | Paired conditional uncertainty, 13 planned comparisons", height=650
    )
    charts.append(fig)
    rows = [r for r in comparisons if r["comparison"].startswith("conditioning:")]
    fig = go.Figure(go.Bar(x=[r["comparison"] for r in rows], y=[r["delta_auc"] for r in rows]))
    fig.add_hline(y=0)
    fig.update_layout(title="03 | Rule conditioning beyond equal-norm shared copies")
    charts.append(fig)
    fig = go.Figure()
    sizes = pd.DataFrame(result["design_sizes"])
    for fold, rows in sizes.groupby("fold"):
        fig.add_bar(x=rows.variant, y=rows["columns"], name=f"Fold {fold}")
    fig.update_layout(
        title="04 | Actual candidate width, including matched zero padding", barmode="group"
    )
    charts.append(fig)
    counts = result["support_coverage"]
    labels = [
        f"Fold {r['fold']} | " + ("advertising" if "advert" in r["policy"] else "legal")
        for r in counts
    ]
    fig = go.Figure()
    for name in ("permitted", "violating"):
        fig.add_bar(x=labels, y=[r[name] for r in counts], name=name)
    fig.update_layout(
        title="05 | Eligible training labels by policy, not query labels", barmode="group"
    )
    charts.append(fig)
    effects = pd.DataFrame(result["dense_rule_effects"])
    if effects.empty:
        fig = go.Figure()
    else:
        means = effects.groupby(["policy", "feature"]).deviation_coefficient.mean().reset_index()
        keep = (
            means.assign(magnitude=means.deviation_coefficient.abs())
            .groupby("feature")
            .magnitude.max()
            .nlargest(12)
            .index
        )
        fig = go.Figure()
        for policy, rows in means[means.feature.isin(keep)].groupby("policy"):
            fig.add_bar(
                y=rows.feature,
                x=rows.deviation_coefficient,
                orientation="h",
                name="Advertising" if "advert" in policy else "Legal advice",
            )
    fig.update_layout(
        title="06 | Fitted dense policy deviations; not causal importance",
        barmode="group",
        height=650,
    )
    charts.append(fig)
    rows = [r for r in comparisons if r["comparison"].startswith("removal:")]
    fig = go.Figure(go.Bar(x=[r["comparison"] for r in rows], y=[r["delta_auc"] for r in rows]))
    fig.add_hline(y=0)
    fig.update_layout(title="07 | Family-removal ablations within the combined representation")
    charts.append(fig)
    fig = go.Figure()
    for field in ("brier", "log_loss"):
        means = metrics.groupby("variant")[field].mean()
        fig.add_bar(x=means.index, y=means.values, name=field)
    fig.update_layout(title="08 | Probability quality: lower is better", barmode="group")
    charts.append(fig)
    for fig in charts:
        fig.update_layout(
            template="plotly_white",
            height=fig.layout.height or 520,
            margin=dict(l=120, r=40, t=90, b=150),
            font=dict(size=13),
            xaxis=dict(automargin=True),
            yaxis=dict(automargin=True),
        )
    return charts


def bounded_compute(root: Path):
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "scripts.run_policy_features",
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
        raise RuntimeError("policy worker stopped; preserve checkpoints and return ZIP")
    return json.loads((root / PUBLIC / "results.json").read_text())


def write_dashboard(root: Path, result):
    import html

    import plotly.io as pio

    title = "Jigsaw | Round 3: policy-specific feature effects"
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
    destination = destination or Path.home() / "jigsaw_feature_round3_return.zip"
    names = [
        "scripts/policy_features.py",
        "scripts/run_policy_features.py",
        CONFIG,
        "tests/test_policy_features.py",
        "docs/POLICY_FEATURES.md",
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
