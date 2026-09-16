"""Frozen-feature combination screen. No cloud calls, downloads, or automatic promotion."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import signal
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock
from scipy import sparse
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from jigsaw_rules.data import EXAMPLES, normalize
from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest
from scripts.run_behavioral_features import stage_read, stage_write, weighted_auc_samples

CONFIG = "configs/feature_combination_evaluation.json"
PUBLIC = "reports/feature_combinations"
PRIVATE = "runs/feature_combinations"
NOTEBOOKS = (
    "notebooks/22_feature_campaign_review.ipynb",
    "notebooks/23_feature_combination_evaluation.ipynb",
)
SOURCE = "scripts/feature_combination_evaluation.py"
GROUPS = (
    "behavior",
    "scope",
    "rule_alignment",
    "relations",
    "legacy_support",
    "local_density",
    "matched_pairs",
    "conditioned_geometry",
    "actor_support",
    "crossmodal",
    "prototypes",
    "consistency",
    "passages",
    "bm25",
    "windows",
    "lexical",
)
STUDIES = (
    "behavioral_features",
    "relational_features",
    "policy_features",
    "evidence_features",
    "scope_lexical_features",
    "local_support_features",
    "matched_support_features",
    "conditioned_geometry_features",
    "behavior_support_features",
    "crossmodal_support_features",
    "multiprototype_features",
    "reference_consistency_features",
    "passage_support_features",
    "bm25_support_features",
    "action_window_features",
)
WARNINGS = [
    "Development screen only: the same 881 comments informed previous research decisions.",
    "Conditional bootstrap intervals do not undo historical selection bias or refit encoders.",
    "Adapted training answer margins are in-sample; frozen-margin sensitivity is separate.",
    "Do not use cached supervised banks as precomputed inputs to a new cross-validation split.",
    "Before promotion, rebuild every supervised transform inside nested group-safe folds.",
    "Only two policies are observed; these results do not estimate unseen-policy performance.",
    "Early sparse variants use new same-rule training fits here, not their old pooled fits.",
    "No new Kaggle score, independent confirmation, or automatic model promotion is produced.",
]


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def plan():
    all_groups = list(GROUPS)
    compact = ["behavior", "relations", "passages", "windows", "lexical"]
    out = []

    def add(name, groups, kind, **kw):
        out.append(
            {
                "name": name,
                "groups": list(groups),
                "kind": kind,
                "C": 1.0,
                "margin": "adapted",
                "lexical_mode": "plain",
                **kw,
            }
        )

    add("all_features", all_groups, "full")
    add("all_features_stronger_penalty", all_groups, "regularization", C=0.1)
    add("anchor_refit", [], "control")
    for group in GROUPS:
        add("add_" + group, [group], "addition")
    for group in GROUPS:
        add("without_" + group, [g for g in GROUPS if g != group], "removal")
    for left, right in (
        ("passages", "windows"),
        ("behavior", "relations"),
        ("passages", "actor_support"),
        ("lexical", "scope"),
        ("bm25", "passages"),
        ("consistency", "actor_support"),
    ):
        add("pair_" + left + "_" + right, [left, right], "pair")
    add("compact", compact, "compact")
    add("compact_stronger_penalty", compact, "regularization", C=0.1)
    add("frozen_margin_anchor", [], "margin_sensitivity", margin="frozen")
    add("frozen_margin_all", all_groups, "margin_sensitivity", margin="frozen")
    add("all_lexical_evidence", all_groups, "encoding", lexical_mode="evidence")
    add("compact_lexical_evidence", compact, "encoding", lexical_mode="evidence")
    add("all_lexical_scope", all_groups, "encoding", lexical_mode="scope")
    add("all_lexical_scope_copy", all_groups, "encoding_control", lexical_mode="copy")
    for index, row in enumerate(out):
        row["batch"] = min(3, index // 17 + 1)
    if len(out) != 49 or len({r["name"] for r in out}) != 49:
        raise AssertionError("locked plan has changed")
    return out


def check_file(path, expected):
    if not path.is_file() or path.is_symlink() or digest(path) != expected:
        raise ValueError("Missing or changed verified input: " + str(path))


def load_reports(root, config):
    reports, summary = {}, []
    for number, slug in enumerate(STUDIES, 1):
        path = root / "reports" / slug / "results.json"
        check_file(path, config["reports"][slug])
        report = json.loads(path.read_text())
        if not str(report["status"]).endswith("COMPLETE"):
            raise ValueError("Prior scientific study is incomplete: " + slug)
        reports[slug] = report
        metrics = pd.DataFrame(report["metrics"])
        for name, rows in metrics.groupby("variant", sort=True):
            if set(rows.fold) != {0, 1} or len(rows) != 2:
                raise ValueError("Prior score alignment differs")
            rows = rows.sort_values("fold")
            summary.append(
                {
                    "round": number,
                    "study": slug,
                    "variant": name,
                    "advertising_auc": float(rows.iloc[0].auc),
                    "legal_advice_auc": float(rows.iloc[1].auc),
                    "macro_auc": float(rows.auc.mean()),
                    "mean_brier": float(rows.brier.mean()),
                    "mean_log_loss": float(rows.log_loss.mean()),
                    "round_decision": report["decision"],
                }
            )
    return reports, summary


def read_arrays(path, ntrain, nquery, width):
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"training", "query"}:
            raise ValueError("Unknown feature-bank schema: " + str(path))
        tx, qx = data["training"].copy(), data["query"].copy()
    if tx.shape != (ntrain, width) or qx.shape != (nquery, width):
        raise ValueError("Feature-bank row/column alignment differs: " + str(path))
    if not np.isfinite(tx).all() or not np.isfinite(qx).all():
        raise ValueError("Nonfinite feature bank")
    return tx, qx


def bank(root, reports, slug, fold, directory, filename, width, ntrain, nquery):
    r = reports[slug]
    folder = root / "runs" / slug / r["run_id"] / f"fold_{fold}" / directory
    key = {"run_id": r["run_id"], "fold": fold}
    if slug == "local_support_features":
        key["feature_bank"] = "frozen"
    else:
        key["stage"] = "paired_features" if slug == "matched_support_features" else "feature_bank"
    if stage_read(folder, key) is None:
        raise ValueError("Completed bank is missing; no old experiment will be rerun")
    for name, expected in r["identity"]["source"].items():
        check_file(root / name, expected)
    return read_arrays(folder / filename, ntrain, nquery, width)


def lexical_blocks(train, query):
    """Existing R1/R4/R5 representations refitted on eligible support rows, never queries."""
    from scripts.evidence_features import RuleEvidence
    from scripts.scope_lexical_features import ScopedLexicon

    def texts(frame):
        return ("RULE: " + frame.rule + "\nBODY: " + frame.body).tolist()

    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=12000, sublinear_tf=True)
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=20000, sublinear_tf=True
    )
    wx, cx = word.fit_transform(texts(train)), char.fit_transform(texts(train))
    wq, cq = word.transform(texts(query)), char.transform(texts(query))
    y, repeats = train.rule_violation.to_numpy(int), train.repeat.to_numpy(float)
    ew = RuleEvidence().fit(wx, y, train.rule, repeats)
    ec = RuleEvidence().fit(cx, y, train.rule, repeats)
    ewx, _ = ew.transform(wx, train.rule, mode="rule")
    ewq, _ = ew.transform(wq, query.rule, mode="rule")
    ecx, _ = ec.transform(cx, train.rule, mode="rule")
    ecq, _ = ec.transform(cq, query.rule, mode="rule")
    result = {
        "plain": (sparse.hstack([wx, cx], format="csr"), sparse.hstack([wq, cq], format="csr")),
        "evidence": (
            sparse.hstack([ewx, ecx], format="csr"),
            sparse.hstack([ewq, ecq], format="csr"),
        ),
    }
    lexicon = ScopedLexicon().fit(train, {"word_features": 12000}, word.get_feature_names_out(), ew)
    sx, _, _ = lexicon.transform(train[["body", "rule"]])
    sq, _, _ = lexicon.transform(query[["body", "rule"]])
    for mode in ("scope", "copy"):
        result[mode] = (
            sparse.hstack([result["evidence"][0], *(sx[f][mode] for f in sx)], format="csr"),
            sparse.hstack([result["evidence"][1], *(sq[f][mode] for f in sq)], format="csr"),
        )
    return result, {"word_columns": wx.shape[1], "character_columns": cx.shape[1]}


def first_submission_count(frame):
    # Counting the historical full-training vocabulary must never feed evaluation features.
    corpus = [text for name in ("body", "rule", *EXAMPLES) for text in frame[name]]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, max_features=40000, dtype=np.float64
    ).fit(corpus)
    return {
        "word_unigram_bigram_columns": len(vectorizer.vocabulary_),
        "explicit_similarity_columns": 8,
        "total_columns": len(vectorizer.vocabulary_) + 8,
        "classifier_fits": 0,
        "purpose": "historical feature count only, not validation",
    }


def prepare(root):
    from scripts import run_action_window_features as last
    from scripts.behavioral_features import basic_features, cross_fitted_support
    from scripts.relational_features import relational_features

    config = json.loads((root / CONFIG).read_text())
    reports, historical = load_reports(root, config)
    # This calls verification/read paths only, not run_study or feature_bank constructors.
    old9, old6, directory, bundles, frame, audit, cohorts = last.verify_prior(
        root, reports["action_window_features"]["identity"]["config"]
    )
    refs, anchors = last.prepare_controls(old6, directory, bundles, frame, root, old9)
    ident = {
        "config": config,
        "source": digest(root / SOURCE),
        "environment": reports["action_window_features"]["identity"]["environment"],
        "cohorts": cohorts,
    }
    run_id = json_hash(ident)[:20]
    work = root / PRIVATE / run_id
    key = {"run_id": run_id, "stage": "prepared_inputs"}
    marker = stage_read(work / "inputs", key)
    if marker is not None:
        saved = json.loads((work / "inputs/summary.json").read_text())
        target = root / PUBLIC / "review.json"
        if target.exists() and json.loads(target.read_text()) != saved:
            raise ValueError("Prepared public review changed; preserve it")
        if not target.exists():
            atomic_json(target, saved)
        return run_id, saved
    # Existing banks share the exact cached training ordering verified by last.verify_prior.
    locators = (
        ("matched_pairs", "matched_support_features", "pair_bank", "matched.npz", 12),
        ("conditioned_geometry", "conditioned_geometry_features", "feature_bank", "dual.npz", 18),
        ("crossmodal", "crossmodal_support_features", "feature_bank", "aligned.npz", 48),
        ("prototypes", "multiprototype_features", "feature_bank", "clustered.npz", 36),
        ("consistency", "reference_consistency_features", "feature_bank", "observed.npz", 48),
        ("passages", "passage_support_features", "feature_bank", "passages.npz", 48),
        ("bm25", "bm25_support_features", "feature_bank", "bm25.npz", 48),
        ("windows", "action_window_features", "feature_bank", "windows.npz", 48),
    )
    files, inventory = {}, []
    group_lookup = {}
    for b in bundles:
        for t in b["query"].body:
            group_lookup.setdefault(normalize(t), len(group_lookup))
    for fold, bundle in enumerate(bundles):
        tr, q = bundle["train"], bundle["query"]
        if set(q) != {"body", "rule", "row_id"}:
            raise ValueError("Evaluation feature input contains targets")
        if set(tr.body.map(normalize)) & set(q.body.map(normalize)):
            raise ValueError("Training/query text overlap")
        blocks = {}
        full = bank(
            root,
            reports,
            "local_support_features",
            fold,
            "features_frozen",
            "features.npz",
            21,
            len(tr),
            len(q),
        )
        blocks["local_density"] = (full[0][:, 9:], full[1][:, 9:])
        behavior = bank(
            root,
            reports,
            "behavior_support_features",
            fold,
            "feature_bank",
            "conditioned.npz",
            48,
            len(tr),
            len(q),
        )
        blocks["actor_support"] = (behavior[0][:, :24], behavior[1][:, :24])
        for name, slug, folder, filename, width in locators:
            blocks[name] = bank(root, reports, slug, fold, folder, filename, width, len(tr), len(q))
        bt, bq = basic_features(tr[["body", "rule"]]), basic_features(q[["body", "rule"]])
        for family in ("behavior", "scope", "rule_alignment"):
            cols = [name for name in bt if name.startswith(family + "/")]
            blocks[family] = (bt[cols].to_numpy(float), bq[cols].to_numpy(float))
        blocks["relations"] = (
            relational_features(tr[["body"]]).to_numpy(float),
            relational_features(q[["body"]]).to_numpy(float),
        )
        st, sq, _ = cross_fitted_support(tr, q)
        blocks["legacy_support"] = st.to_numpy(float), sq.to_numpy(float)
        lexical, words = lexical_blocks(tr, q)
        arrays = {
            "base_training": anchors[fold]["training"],
            "base_query": anchors[fold]["query"],
            "frozen_training": bundle["arrays"]["train_frozen_scores"][:, 1],
            "frozen_query": bundle["arrays"]["query_frozen_scores"][:, 1],
            "train_y": tr.rule_violation.to_numpy(int),
            "repeat": tr.repeat.to_numpy(float),
            "query_y": frame.set_index("row_id").loc[q.row_id].rule_violation.to_numpy(int),
            "row_ids": q.row_id.to_numpy(),
            "bootstrap_groups": np.asarray([group_lookup[normalize(t)] for t in q.body]),
        }
        for name, (tx, qx) in blocks.items():
            if tx.shape != (len(tr), qx.shape[1]) or len(qx) != len(q):
                raise ValueError("Incompatible feature-block alignment: " + name)
            arrays[name + "_training"], arrays[name + "_query"] = tx, qx
            inventory.append(
                {
                    "fold": fold,
                    "family": name,
                    "columns": tx.shape[1],
                    "training_rows": len(tr),
                    "query_rows": len(q),
                }
            )
        for name in last.CONTROLS:
            arrays["control_" + name] = refs[(fold, name)]["score"]
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        files[f"fold_{fold}.npz"] = buffer.getvalue()
        for mode, matrices in lexical.items():
            for side, matrix in zip(("training", "query"), matrices, strict=True):
                buffer = io.BytesIO()
                sparse.save_npz(buffer, matrix)
                files[f"fold_{fold}_lexical_{mode}_{side}.npz"] = buffer.getvalue()
        inventory.append(
            {
                "fold": fold,
                "family": "lexical",
                "columns": lexical["plain"][0].shape[1],
                "training_rows": len(tr),
                "query_rows": len(q),
                **words,
            }
        )
    summary = {
        "run_id": run_id,
        "identity": ident,
        "inventory": inventory,
        "historical": historical,
        "first_submission": first_submission_count(frame),
        "plan": plan(),
        "total_new_model_fits": 98,
        "new_feature_families": 0,
        "limitations": WARNINGS,
        "group_count": len(group_lookup),
    }
    files["summary.json"] = json.dumps(summary, indent=2).encode()
    stage_write(work / "inputs", files, key)
    public = root / PUBLIC
    atomic_json(public / "review.json", summary)
    return run_id, summary


def dense_transform(tx, qx):
    """Train-only constant/exact-duplicate removal and standardization."""
    tx, qx = np.asarray(tx, float), np.asarray(qx, float)
    if tx.ndim != 2 or qx.ndim != 2 or tx.shape[1] != qx.shape[1]:
        raise ValueError("Dense matrix alignment differs")
    if not np.isfinite(tx).all() or not np.isfinite(qx).all():
        raise ValueError("Nonfinite dense matrix")
    keep, seen = [], set()
    for j in np.flatnonzero(np.var(tx, axis=0) > 1e-12):
        column = tx[:, j].copy()
        column[column == 0] = 0.0
        fingerprint = hashlib.sha256(column.tobytes()).hexdigest()
        if fingerprint not in seen:
            keep.append(int(j))
            seen.add(fingerprint)
    if not keep:
        raise ValueError("No variable training columns")
    scaler = StandardScaler().fit(tx[:, keep])
    return (
        scaler.transform(tx[:, keep]),
        scaler.transform(qx[:, keep]),
        {
            "keep": keep,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "dense_input_columns": tx.shape[1],
            "dense_retained_columns": len(keep),
        },
    )


def build_design(data, inputs, fold, candidate):
    tx, qx = data["base_training"].copy(), data["base_query"].copy()
    if candidate["margin"] == "frozen":
        tx[:, 0], qx[:, 0] = data["frozen_training"], data["frozen_query"]
    groups = candidate["groups"]
    for group in groups:
        if group != "lexical":
            tx = np.column_stack([tx, data[group + "_training"]])
            qx = np.column_stack([qx, data[group + "_query"]])
    tx, qx, info = dense_transform(tx, qx)
    x, q = sparse.csr_matrix(tx), sparse.csr_matrix(qx)
    info["sparse_columns"] = 0
    if "lexical" in groups:
        mode = candidate["lexical_mode"]
        lx = sparse.load_npz(inputs / f"fold_{fold}_lexical_{mode}_training.npz")
        lq = sparse.load_npz(inputs / f"fold_{fold}_lexical_{mode}_query.npz")
        if lx.shape != (len(tx), lq.shape[1]) or len(qx) != lq.shape[0]:
            raise ValueError("Lexical alignment differs")
        # Vocabulary is fitted only on this support training population.
        present = np.asarray(lx.power(2).sum(axis=0)).ravel() > 0
        x, q = (
            sparse.hstack([x, lx[:, present]], format="csr"),
            sparse.hstack([q, lq[:, present]], format="csr"),
        )
        info["sparse_columns"] = int(present.sum())
    if not np.isfinite(x.data).all() or not np.isfinite(q.data).all():
        raise ValueError("Nonfinite final matrix")
    return x, q, info


def fit_model(x, y, q, weights, candidate):
    if len(y) != x.shape[0] or set(np.unique(y)) != {0, 1}:
        raise ValueError("Training targets must be aligned and binary")
    if np.shape(weights) != np.shape(y) or not np.isfinite(weights).all() or min(weights) <= 0:
        raise ValueError("Invalid training weights")
    model = LogisticRegression(
        C=candidate["C"], solver="liblinear", max_iter=2000, random_state=2025
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y, sample_weight=weights)
    score = model.decision_function(q)
    return {
        "score": score,
        "probability": expit(score),
        "coefficients": model.coef_.ravel(),
        "intercept": model.intercept_,
    }


def load_prepared(root):
    public = root / PUBLIC
    summary = json.loads((public / "review.json").read_text())
    run_id = summary["run_id"]
    if json_hash(summary["identity"])[:20] != run_id:
        raise ValueError("Prepared identity differs")
    if digest(root / SOURCE) != summary["identity"]["source"]:
        raise ValueError("Evaluation source changed; no silent restart")
    if json.loads((root / CONFIG).read_text()) != summary["identity"]["config"]:
        raise ValueError("Evaluation plan/configuration changed")
    inputs = root / PRIVATE / run_id / "inputs"
    if stage_read(inputs, {"run_id": run_id, "stage": "prepared_inputs"}) is None:
        raise ValueError("Prepared inputs missing")
    if json.loads((inputs / "summary.json").read_text()) != summary:
        raise ValueError("Public/private review differs")
    from jigsaw_rules.runtime import environment

    if environment() != summary["identity"]["environment"]:
        raise ValueError("Environment changed after preparation")
    load_reports(root, summary["identity"]["config"])
    return run_id, summary, inputs


def run_batch(root, batch, max_new_fits=None):
    run_id, summary, inputs = load_prepared(root)
    jobs = [c for c in plan() if c["batch"] == batch]
    if not jobs:
        raise ValueError("Batch must be 1, 2, or 3")
    work = root / PRIVATE / run_id
    count = reused = 0
    with (
        threadpool_limits(limits=1),
        Progress(work / "events.jsonl", f"combination_batch_{batch}", heartbeat_seconds=15) as log,
    ):
        for fold in range(2):
            with np.load(inputs / f"fold_{fold}.npz", allow_pickle=False) as z:
                data = {n: z[n].copy() for n in z.files}
            for candidate in jobs:
                path = work / f"fold_{fold}" / candidate["name"]
                key = {"run_id": run_id, "fold": fold, "candidate": candidate}
                if stage_read(path, key) is not None:
                    reused += 1
                    continue
                if max_new_fits is not None and count >= max_new_fits:
                    raise TimeoutError("Authored interruption; valid fits retained")
                x, q, info = build_design(data, inputs, fold, candidate)
                result = fit_model(x, data["train_y"], q, data["repeat"], candidate)
                buffer = io.BytesIO()
                np.savez_compressed(buffer, row_ids=data["row_ids"], **result)
                stage_write(
                    path,
                    {
                        "predictions.npz": buffer.getvalue(),
                        "preprocessing.json": json.dumps(info).encode(),
                    },
                    key,
                )
                count += 1
                log.emit(
                    "candidate_complete",
                    fold=fold,
                    candidate=candidate["name"],
                    completed=count + reused,
                    total=2 * len(jobs),
                    new_fits=count,
                )
        atomic_json(
            work / f"batch_{batch}.json",
            {"batch": batch, "new_fits": count, "reused_fits": reused, "status": "BATCH_COMPLETE"},
        )
    return summarize(root)


def metrics(y, score):
    probability = expit(score)
    return {
        "auc": float(roc_auc_score(y, score)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
    }


def summarize(root):
    run_id, review, inputs = load_prepared(root)
    config = review["identity"]["config"]
    work = root / PRIVATE / run_id
    data = []
    for fold in range(2):
        with np.load(inputs / f"fold_{fold}.npz", allow_pickle=False) as z:
            data.append({n: z[n].copy() for n in z.files})
    scores = {}
    for name in ("qwen_raw", "answer_only", "frozen_basic", "context_evidence", "uniform_all"):
        scores[name] = [d["control_" + name] for d in data]
    shapes, complete = [], []
    for candidate in plan():
        pair = []
        for fold in range(2):
            path = work / f"fold_{fold}" / candidate["name"]
            key = {"run_id": run_id, "fold": fold, "candidate": candidate}
            if stage_read(path, key) is None:
                break
            with np.load(path / "predictions.npz", allow_pickle=False) as z:
                if not np.array_equal(z["row_ids"], data[fold]["row_ids"]):
                    raise ValueError("Prediction row order differs")
                score = z["score"].copy()
                if score.shape != data[fold]["row_ids"].shape or not np.isfinite(score).all():
                    raise ValueError("Invalid saved score")
                if not np.allclose(expit(score), z["probability"], atol=1e-12, rtol=0):
                    raise ValueError("Score/probability mismatch")
                pair.append(score)
            dims = json.loads((path / "preprocessing.json").read_text())
            shapes.append(
                {
                    "fold": fold,
                    "variant": candidate["name"],
                    **{
                        k: dims[k]
                        for k in ("dense_input_columns", "dense_retained_columns", "sparse_columns")
                    },
                }
            )
        if len(pair) == 2:
            scores[candidate["name"]] = pair
            complete.append(candidate["name"])
    rows, per_policy, draws = [], [], {}
    weights = np.random.default_rng(config["seed"]).multinomial(
        review["group_count"],
        np.full(review["group_count"], 1 / review["group_count"]),
        size=config["bootstrap_replicates"],
    )
    for name, pair in scores.items():
        per, samples = [], []
        for fold, score in enumerate(pair):
            y = data[fold]["query_y"]
            per.append(metrics(y, score))
            per_policy.append({"variant": name, "fold": fold, **per[-1]})
            sample_weights = weights[:, data[fold]["bootstrap_groups"]]
            samples.append(weighted_auc_samples(y, score, sample_weights))
        ys = np.concatenate([d["query_y"] for d in data])
        points = np.concatenate(pair)
        ranked = np.concatenate([(rankdata(x) - 0.5) / len(x) for x in pair])
        rows.append(
            {
                "variant": name,
                "advertising_auc": per[0]["auc"],
                "legal_advice_auc": per[1]["auc"],
                "macro_auc": (per[0]["auc"] + per[1]["auc"]) / 2,
                "pooled_auc": float(roc_auc_score(ys, expit(points))),
                "ranked_pooled_auc": float(roc_auc_score(ys, ranked)),
                "mean_brier": np.mean([m["brier"] for m in per]),
                "mean_log_loss": np.mean([m["log_loss"] for m in per]),
            }
        )
        draws[name] = np.mean(samples, axis=0)
    means = {r["variant"]: r["macro_auc"] for r in rows}
    contrasts = []
    for name in complete:
        for baseline in ("qwen_raw", "context_evidence"):
            contrasts.append((name, baseline, name + " vs " + baseline))
    if "all_features" in scores:
        for g in GROUPS:
            if "without_" + g in scores:
                contrasts.append(("all_features", "without_" + g, "full-minus-family: " + g))
    if "all_lexical_scope" in scores and "all_lexical_scope_copy" in scores:
        contrasts.append(("all_lexical_scope", "all_lexical_scope_copy", "scope vs matched copy"))
    intervals = []
    # Centered max-deviation bands cover completed contrasts; partial bands are provisional.
    if contrasts:
        delta = np.array([means[a] - means[b] for a, b, _ in contrasts])
        matrix = np.column_stack([draws[a] - draws[b] for a, b, _ in contrasts])
        valid = np.isfinite(matrix).all(axis=1)
        if valid.sum() < 0.9 * len(matrix):
            raise ValueError("Too few valid group bootstrap draws")
        band = float(np.quantile(np.max(np.abs(matrix[valid] - delta), axis=1), 0.95))
        for (a, b, title), value in zip(contrasts, delta, strict=True):
            intervals.append(
                {
                    "candidate": a,
                    "reference": b,
                    "comparison": title,
                    "delta": float(value),
                    "low": float(value - band),
                    "high": float(value + band),
                }
            )
    result = {
        "status": "COMBINATION_SCREEN_COMPLETE" if len(complete) == 49 else "PARTIAL_SCREEN",
        "run_id": run_id,
        "models": rows,
        "per_policy": per_policy,
        "comparisons": intervals,
        "design_sizes": shapes,
        "completed_variants": complete,
        "planned_variants": 49,
        "model_fits_completed": 2 * len(complete),
        "new_feature_families": 0,
        "new_neural_inference": 0,
        "selection": "NONE_AUTOMATIC_DEVELOPMENT_ONLY",
        "limitations": WARNINGS,
    }
    atomic_json(root / PUBLIC / "results.json", result)
    return result


def review_figures(review):
    import plotly.graph_objects as go

    table = pd.DataFrame(review["historical"])
    recent = table[table["round"] >= 6]
    best = (
        recent.sort_values("macro_auc", ascending=False)
        .drop_duplicates(["advertising_auc", "legal_advice_auc"])
        .head(16)
    )
    charts = []
    f = go.Figure()
    for field in ("advertising_auc", "legal_advice_auc"):
        f.add_bar(x=best.study + "/" + best.variant, y=best[field], name=field)
    f.update_layout(title="01 | Saved same-cohort readouts: not Kaggle scores", barmode="group")
    charts.append(f)
    counts = pd.DataFrame(review.get("inventory", []))
    f = go.Figure()
    if not counts.empty:
        for fold, part in counts.groupby("fold"):
            dense = part[part.family != "lexical"]
            f.add_bar(x=dense.family, y=dense["columns"], name=f"Policy {fold}")
    f.update_layout(title="02 | Candidate bank dimensions before training-only pruning")
    charts.append(f)
    f = go.Figure(
        go.Scatter(
            x=best.advertising_auc,
            y=best.legal_advice_auc,
            text=best.study + "/" + best.variant,
            mode="markers",
        )
    )
    f.update_layout(
        title="03 | Policy trade-offs",
        xaxis_title="Advertising AUC",
        yaxis_title="Legal-advice AUC",
    )
    charts.append(f)
    f = go.Figure(
        go.Bar(x=table.groupby("round").size().index, y=table.groupby("round").size().values)
    )
    f.update_layout(title="04 | Recorded readouts per round, including reused controls")
    charts.append(f)
    for j, field in enumerate(("mean_brier", "mean_log_loss"), 5):
        f = go.Figure(go.Bar(x=best.study + "/" + best.variant, y=best[field]))
        f.update_layout(title=f"{j:02d} | {field}: lower is better")
        charts.append(f)
    return charts


def figures(result):
    import plotly.graph_objects as go

    models = pd.DataFrame(result["models"])
    charts = []
    f = go.Figure()
    for field in ("advertising_auc", "legal_advice_auc"):
        f.add_bar(x=models.variant, y=models[field], name=field)
    f.update_layout(title="01 | Same-row policy AUC", barmode="group")
    charts.append(f)
    f = go.Figure()
    if result["comparisons"]:
        part = pd.DataFrame(result["comparisons"])
        part = part[part.reference == "context_evidence"]
        f.add_scatter(
            x=part.delta,
            y=part.candidate,
            mode="markers",
            error_x={"array": part.high - part.delta},
        )
    f.add_vline(x=0)
    f.update_layout(
        title="02 | Conditional simultaneous intervals vs context reference", height=800
    )
    charts.append(f)
    for prefix, title in (
        ("add_", "03 | Add-one-family scores"),
        ("without_", "04 | Leave-one-family-out scores"),
    ):
        part = models[models.variant.str.startswith(prefix)]
        f = go.Figure(go.Bar(x=part.variant, y=part.macro_auc))
        f.update_layout(title=title)
        charts.append(f)
    part = models[models.variant.str.startswith(("pair_", "compact"))]
    f = go.Figure(go.Bar(x=part.variant, y=part.macro_auc))
    f.update_layout(title="05 | Prespecified combinations, not a validation-tuned ensemble")
    charts.append(f)
    f = go.Figure()
    for field in ("macro_auc", "pooled_auc", "ranked_pooled_auc"):
        f.add_bar(x=models.variant, y=models[field], name=field)
    f.update_layout(title="06 | Three distinct AUC summaries", barmode="group")
    charts.append(f)
    for j, field in enumerate(("mean_brier", "mean_log_loss"), 7):
        f = go.Figure(go.Bar(x=models.variant, y=models[field]))
        f.update_layout(title=f"{j:02d} | {field}: lower is better")
        charts.append(f)
    f = go.Figure()
    if result["design_sizes"]:
        sizes = pd.DataFrame(result["design_sizes"])
        for fold, part in sizes.groupby("fold"):
            f.add_bar(
                x=part.variant,
                y=part.dense_retained_columns + part.sparse_columns,
                name=f"Policy {fold}",
            )
    f.update_layout(title="09 | Actual retained columns, not nominal feature counts")
    charts.append(f)
    part = models[models.variant.str.contains("frozen|penalty|lexical|all_features")]
    f = go.Figure(go.Bar(x=part.variant, y=part.macro_auc))
    f.update_layout(title="10 | Encoding, penalty, and frozen-margin sensitivity")
    charts.append(f)
    for chart in charts:
        chart.update_layout(
            height=chart.layout.height or 550,
            margin={"b": 170},
            xaxis={"automargin": True},
            yaxis={"automargin": True},
        )
    return charts


def render(root, comparison=False):
    import nbformat
    import plotly.io as pio
    from nbclient import NotebookClient

    public, private = root / PUBLIC, root / PRIVATE
    path = root / NOTEBOOKS[int(comparison)]
    source = path.read_bytes()
    nb = nbformat.reads(source.decode(), as_version=4)
    source_hash = json_hash([(c.cell_type, c.source) for c in nb.cells])
    artifact = public / ("results.json" if comparison else "review.json")
    key = {"source": source_hash, "input_sha256": digest(artifact)}
    marker = private / ("comparison_notebook.json" if comparison else "review_notebook.json")
    if marker.exists():
        prior = json.loads(marker.read_text())
        if prior["key"] == key and prior["notebook_sha256"] == digest(path):
            return {**prior, "cells_executed": 0, "status": "NOTEBOOK_REUSED"}
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
    NotebookClient(
        nb, timeout=60, kernel_name="jigsaw-rules", resources={"metadata": {"path": str(root)}}
    ).execute()
    cells = [c for c in nb.cells if c.cell_type == "code"]
    count = sum(
        "application/vnd.plotly.v1+json" in o.get("data", {}) for c in cells for o in c.outputs
    )
    expected = 10 if comparison else 6
    if any(c.execution_count is None for c in cells) or count != expected:
        raise ValueError("Notebook output contract failed")
    if path.read_bytes() != source:
        raise ValueError("Notebook changed concurrently; executed copy was not substituted")
    atomic_bytes(path, nbformat.writes(nb).encode())
    record = {
        "key": key,
        "notebook_sha256": digest(path),
        "cells_executed": len(cells),
        "plotly_charts": count,
        "status": "NOTEBOOK_EXECUTED",
    }
    atomic_json(marker, record)
    value = json.loads(artifact.read_text())
    plots = figures(value) if comparison else review_figures(value)
    text = [
        "<!doctype html><meta charset='utf-8'><h1>Jigsaw feature evaluation</h1>",
        "<p>Previously inspected development data. No automatic promotion or Kaggle score.</p>",
    ]
    for i, plot in enumerate(plots):
        text.append(pio.to_html(plot, full_html=False, include_plotlyjs=(i == 0)))
    atomic_bytes(
        public / ("dashboard.html" if comparison else "review_dashboard.html"),
        "\n".join(text).encode(),
    )
    return record


def export(root, destination=None):
    destination = destination or Path.home() / "jigsaw_feature_combinations_return.zip"
    names = [
        SOURCE,
        CONFIG,
        "docs/FEATURE_COMBINATION_EVALUATION.md",
        "tests/test_feature_combination_evaluation.py",
        *NOTEBOOKS,
    ]
    names += [PUBLIC + "/" + n for n in ("review.json", "results.json")]
    names += [
        PRIVATE + "/" + n
        for n in (
            "launcher_report.json",
            "tests.xml",
            "review_notebook.json",
            "comparison_notebook.json",
        )
    ]
    payload = {n: (root / n).read_bytes() for n in names if (root / n).is_file()}
    payload["SHA256SUMS.json"] = json.dumps(
        {n: hashlib.sha256(b).hexdigest() for n, b in payload.items()}, indent=2
    ).encode()
    tmp = destination.with_suffix(".zip.partial")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in payload.items():
            archive.writestr(name, content)
    os.replace(tmp, destination)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--batch", type=int, choices=(1, 2, 3))
    group.add_argument("--render-review", action="store_true")
    group.add_argument("--render-comparison", action="store_true")
    group.add_argument("--export", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.export:
        print("RETURN_FILE:", export(root))
        return
    budget = 240 if args.prepare or args.batch else 110

    def expired(_signum, _frame):
        raise TimeoutError("Stage hard limit reached; completed checkpoints preserved")

    old = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, budget)
    (root / PRIVATE).mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(root / PRIVATE / "evaluation.lock"), timeout=1):
            if args.prepare:
                key, review = prepare(root)
                print("REVIEW_READY:", key)
                print("FIRST_SUBMISSION_FEATURES:", review["first_submission"])
            elif args.batch:
                result = run_batch(root, args.batch)
                print("RESULT:", result["status"])
                print("COMPLETED_VARIANTS:", len(result["completed_variants"]), "/ 49")
            else:
                comparison = args.render_comparison
                first = render(root, comparison)
                second = render(root, comparison)
                if second["cells_executed"] != 0:
                    raise ValueError("Notebook replay executed cells")
                print("NOTEBOOK:", first["status"], "PLOTLY:", first["plotly_charts"])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


if __name__ == "__main__":
    main()
