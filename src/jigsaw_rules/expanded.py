"""Four-policy development study with no confirmation-target access or promotion."""

from __future__ import annotations

import json
import warnings
from itertools import combinations, product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from filelock import FileLock
from scipy import sparse
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from jigsaw_rules.data import EXAMPLES, normalize, validate_frame
from jigsaw_rules.diagnostics import FIT_VARIANTS, fit_lexical, support_scores
from jigsaw_rules.embeddings import content_key
from jigsaw_rules.expanded_embeddings import extend_embeddings
from jigsaw_rules.metrics import evaluate
from jigsaw_rules.model import LexicalClassifier
from jigsaw_rules.released import load_plan as released_plan
from jigsaw_rules.released import load_research, released_evidence
from jigsaw_rules.representations import semantic_candidates, structural_candidates
from jigsaw_rules.research import BUDGETS, CORE, VARIANTS, _model, cached_vectors, fold_bank
from jigsaw_rules.robustness import POLICY, purge_near
from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest, environment, stage
from jigsaw_rules.splits import make_splits
from jigsaw_rules.uncertainty import paired_auc_comparisons

PROTOCOL_COMMIT = "5c16f91c04880d52280bf5a0a71ff688b327bb70"
PROTOCOL_SHA256 = "8781727105c67dd18a19262203650e22de9b9009ba8518a325cb39ed49825356"
PUBLIC_FILES = (
    "results.json",
    "uncertainty.json",
    "screening.json",
    "importance.json",
    "audit.json",
)
SOURCES = (
    "expanded.py",
    "expanded_embeddings.py",
    "research.py",
    "representations.py",
    "context.py",
    "data.py",
    "diagnostics.py",
    "embeddings.py",
    "semantic.py",
    "metrics.py",
    "model.py",
    "released.py",
    "robustness.py",
    "runtime.py",
    "splits.py",
    "uncertainty.py",
)


def load_plan(root: Path) -> dict:
    path = root / "configs/expanded.json"
    if digest(path) != PROTOCOL_SHA256:
        raise ValueError("Study specification differs from the committed pre-score protocol")
    plan = json.loads(path.read_text())
    if (
        plan["broad_variants"] != len(VARIANTS)
        or plan["representation_controls"] != list(FIT_VARIANTS)
        or plan["near_copy_policy"] != POLICY
    ):
        raise ValueError("Study implementations differ from the frozen comparison plan")
    return plan


def load_development(root: Path) -> pd.DataFrame:
    """Only the verified research export and original training CSV are read."""
    plan = load_plan(root)
    boundary = released_evidence(root)
    if boundary is None or boundary["metadata"]["run_id"] != plan["boundary_run"]:
        raise ValueError("Unexpected released-data boundary")
    old = root / "data/raw/train.csv"
    source = released_plan(root)
    if digest(old) != source["source"]["files"]["train.csv"]["sha256"]:
        raise ValueError("Historical training source differs")
    frame = pd.concat([pd.read_csv(old), load_research(root)], ignore_index=True)
    validate_frame(frame, train=True)
    permitted = {source["rule_text"][name] for name in source["research_rule_ids"]}
    if len(frame) != plan["development_rows"] or set(frame.rule) != permitted:
        raise ValueError("Development cohort has unexpected rows or policies")
    return frame


def conflict_mask(frame: pd.DataFrame) -> np.ndarray:
    keys = pd.DataFrame({"body": frame.body.map(normalize), "rule": frame.rule})
    keys["label"] = frame.rule_violation
    return keys.groupby(["body", "rule"]).label.transform("nunique").gt(1).to_numpy()


def weighted_metrics(frame: pd.DataFrame, probabilities: np.ndarray) -> dict:
    """Each distinct body/policy receives equal total weight, without dropping context."""
    keys = pd.DataFrame({"body": frame.body.map(normalize), "rule": frame.rule})
    weights = 1 / keys.groupby(["body", "rule"]).body.transform("size").to_numpy()
    per_rule = {
        rule: float(
            roc_auc_score(
                group.rule_violation, probabilities[indices], sample_weight=weights[indices]
            )
        )
        for rule, indices in frame.reset_index(drop=True).groupby("rule").indices.items()
        for group in [frame.iloc[indices]]
    }
    return {
        "rule_macro_auc": float(np.mean(list(per_rule.values()))),
        "per_rule_auc": per_rule,
        "log_loss": float(log_loss(frame.rule_violation, probabilities, sample_weight=weights)),
        "brier": float(
            brier_score_loss(frame.rule_violation, probabilities, sample_weight=weights)
        ),
        "distinct_body_policy_groups": int(weights.sum().round()),
    }


def validate_splits(frame: pd.DataFrame, records: dict) -> None:
    expected = set(range(len(frame)))
    for protocol, folds in records.items():
        seen, body_folds = [], {}
        for fold, assignment in enumerate(folds):
            ti, vi = assignment["train"], assignment["valid"]
            if (
                len(ti) != len(set(ti))
                or len(vi) != len(set(vi))
                or set(ti) & set(vi)
                or not set(ti + vi).issubset(expected)
            ):
                raise ValueError("Invalid split row coverage or overlap")
            training, validation = frame.iloc[ti], frame.iloc[vi]
            forbidden = set(validation.body.map(normalize))
            if any(set(training[c].map(normalize)) & forbidden for c in ["body", *EXAMPLES]):
                raise ValueError("Validation body leaks through training text or support")
            if protocol == "heldout_rule" and set(training.rule) & set(validation.rule):
                raise ValueError("Held-out policy appears in training")
            if protocol == "seen_rule":
                for body in forbidden:
                    if body in body_folds and body_folds[body] != fold:
                        raise ValueError("Normalized comment crosses validation folds")
                    body_folds[body] = fold
            seen.extend(vi)
        if len(seen) != len(frame) or set(seen) != expected:
            raise ValueError("OOF assignments must cover each row exactly once")


def design(frame: pd.DataFrame, plan: dict) -> dict:
    records = {
        protocol: [
            {"train": ti.tolist(), "valid": vi.tolist(), "purged": purged}
            for ti, vi, purged in make_splits(
                frame, protocol, plan["seen_rule_folds"], plan["seed"]
            )
        ]
        for protocol in ("seen_rule", "heldout_rule")
    }
    validate_splits(frame, records)
    return records


def filter_training(training: pd.DataFrame, validation: pd.DataFrame, sensitivity: str):
    if sensitivity == "exclude_conflicts":
        # Never use validation labels to infer a training exclusion.
        return training.iloc[np.flatnonzero(~conflict_mask(training))]
    if sensitivity == "purge_near_copies":
        return purge_near(training, validation)[0]
    raise ValueError("Unknown training sensitivity")


def _reference(path: Path, training: pd.DataFrame, validation: pd.DataFrame, name: str) -> None:
    model = LexicalClassifier(context=name == "rule_examples").fit(training)
    part = validation[["row_id", "rule", "rule_violation"]].copy()
    part["probability"] = model.predict(validation)
    atomic_bytes(path / "predictions.csv", part.to_csv(index=False).encode())
    joblib.dump(model, path / "model.joblib", compress=3)
    atomic_json(path / "details.json", {"training_rows": len(training), "context": model.context})


def _matrices(training, validation, word, character, vectors, ti, vi) -> dict:
    direction = vectors[:, 1:3].mean(1) - vectors[:, 3:5].mean(1)
    return {
        "word": (word.transform(training.body), word.transform(validation.body)),
        "character": (character.transform(training.body), character.transform(validation.body)),
        "qwen_body": (sparse.csr_matrix(vectors[ti, 0]), sparse.csr_matrix(vectors[vi, 0])),
        "qwen_interaction": (
            sparse.csr_matrix(vectors[ti, 0] * direction[ti]),
            sparse.csr_matrix(vectors[vi, 0] * direction[vi]),
        ),
    }


def _sensitivity_fit(path, training, validation, name, primary, protocol, fold) -> None:
    if len(training) == json.loads((primary / "details.json").read_text())["training_rows"]:
        atomic_bytes(path / "predictions.csv", (primary / "predictions.csv").read_bytes())
        atomic_json(
            path / "reuse.json",
            {
                "primary": primary.name,
                "reason": "No rows removed",
                "prediction_sha256": digest(primary / "predictions.csv"),
            },
        )
    elif name == "rule_examples":
        _reference(path, training, validation, name)
    else:
        # Refit vocabulary as well as classifier on the reduced training set.
        from jigsaw_rules.research import _vocabulary

        _vocabulary(path, training)
        _, character = joblib.load(path / "vocabulary.joblib")
        matrices = {
            "character": (character.transform(training.body), character.transform(validation.body))
        }
        fit_lexical(path, name, matrices, training, validation, protocol, fold)


def support_stress(frame: pd.DataFrame, vectors: np.ndarray, seed: int) -> dict:
    similarity = np.einsum("nd,nkd->nk", vectors[:, 0], vectors[:, 1:])
    single = np.mean(
        [
            expit(10 * (similarity[:, p] - similarity[:, n]))
            for p, n in product(range(2), range(2, 4))
        ],
        axis=0,
    )
    shuffled = vectors.copy()
    rng = np.random.default_rng(seed)
    for indices in frame.reset_index(drop=True).groupby("rule").indices.values():
        shuffled[indices, 1:] = vectors[rng.permutation(indices), 1:]
    self_match = np.column_stack(
        [frame.body.map(normalize).eq(frame[c].map(normalize)) for c in EXAMPLES]
    ).any(axis=1)
    centroid = support_scores(vectors)["qwen_centroid"]
    return {
        "one_example_per_class": evaluate(frame.rule_violation, single, frame.rule),
        "within_policy_context_shuffle": evaluate(
            frame.rule_violation, support_scores(shuffled)["qwen_centroid"], frame.rule
        ),
        "exclude_self_matches": evaluate(
            frame.loc[~self_match, "rule_violation"],
            centroid[~self_match],
            frame.loc[~self_match, "rule"],
        ),
        "self_match_rows": int(self_match.sum()),
        "seed": seed,
        "interpretation": "Fixed support-score diagnostics; no parameter or threshold selected.",
    }


def aligned_predictions(frame: pd.DataFrame, predictions: pd.DataFrame) -> np.ndarray:
    if predictions.row_id.duplicated().any() or set(predictions.row_id) != set(frame.row_id):
        raise ValueError("Expected exactly one prediction per development row")
    part = predictions.set_index("row_id").loc[frame.row_id]
    if not np.array_equal(part.rule_violation, frame.rule_violation) or not np.array_equal(
        part.rule, frame.rule
    ):
        raise ValueError("Prediction labels or policies differ from development lineage")
    p = part.probability.to_numpy()
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid development probabilities")
    return p


def validate_oof(frame: pd.DataFrame, oof: pd.DataFrame, plan: dict, splits: dict) -> None:
    names = set(plan["lexical_references"]) | set(VARIANTS) | set(plan["representation_controls"])
    names.update(plan["fixed_support_scores"])
    names.update(
        f"{name}_{sensitivity}"
        for name in plan["sensitivity_refits"]
        for sensitivity in plan["sensitivity_training_filters"]
    )
    if set(oof.protocol) != set(splits):
        raise ValueError("OOF protocols differ from the declared design")
    for protocol, records in splits.items():
        subset = oof[oof.protocol == protocol]
        if set(subset.model) != names:
            raise ValueError("OOF model coverage differs from the declared study")
        assignment = {
            frame.iloc[i].row_id: fold
            for fold, record in enumerate(records)
            for i in record["valid"]
        }
        for _, part in subset.groupby("model"):
            aligned_predictions(frame, part)
            if not np.array_equal(part.fold, part.row_id.map(assignment)):
                raise ValueError("OOF fold differs from the saved validation assignment")


def summarize(
    frame, oof, splits, screens, importance, selections, folds, filters, vectors, plan, cache
):
    results, uncertainty, stability = [], [], []
    conflicts = conflict_mask(frame)
    for protocol in splits:
        subset = oof[oof.protocol == protocol]
        bank = {name: aligned_predictions(frame, part) for name, part in subset.groupby("model")}
        for name, probability in bank.items():
            results.append(
                {
                    "protocol": protocol,
                    "model": name,
                    "metrics": evaluate(frame.rule_violation, probability, frame.rule),
                    "equal_body_policy_weight": weighted_metrics(frame, probability),
                    "excluding_conflicts": evaluate(
                        frame.loc[~conflicts, "rule_violation"],
                        probability[~conflicts],
                        frame.loc[~conflicts, "rule"],
                    ),
                }
            )
        contrasts = [(name, name, "rule_examples") for name in bank if name != "rule_examples"]
        contrasts += [
            (f"add_{family}", f"word_{family}", "word_screened")
            for family in BUDGETS
            if family != "word"
        ]
        contrasts += [
            (f"remove_{family}", "all_transfer", f"without_{family}")
            for family in CORE
            if family != "word"
        ]
        uncertainty.extend(
            {"protocol": protocol, **item}
            for item in paired_auc_comparisons(
                frame.rule_violation,
                bank,
                frame.rule,
                frame.body.map(normalize),
                contrasts,
                draws=plan["bootstrap_draws"],
                seed=plan["seed"],
            )
        )
        for family in BUDGETS:
            for first, second in combinations(range(len(splits[protocol])), 2):
                left, right = (
                    selections[(protocol, first, family)],
                    selections[(protocol, second, family)],
                )
                stability.append(
                    {
                        "protocol": protocol,
                        "family": family,
                        "fold_a": first,
                        "fold_b": second,
                        "retained_jaccard": len(left & right) / max(1, len(left | right)),
                    }
                )
    keys = pd.DataFrame({"body": frame.body.map(normalize), "rule": frame.rule})
    fold_audit = [
        {
            "protocol": protocol,
            "fold": fold,
            "training_rows": len(a["train"]),
            "validation_rows": len(a["valid"]),
            "purged_training_rows": a["purged"],
            "validation_policies": sorted(frame.iloc[a["valid"]].rule.unique()),
        }
        for protocol, records in splits.items()
        for fold, a in enumerate(records)
    ]
    return {
        "results.json": results,
        "uncertainty.json": uncertainty,
        "screening.json": {
            "families": screens,
            "stability": stability,
            "model_folds": folds,
            "budgets": BUDGETS,
            "variants": VARIANTS,
        },
        "importance.json": importance,
        "audit.json": {
            "development_rows": len(frame),
            "development_policies": frame.rule.nunique(),
            "class_counts": frame.groupby("rule")
            .rule_violation.agg(["size", "sum"])
            .reset_index()
            .to_dict("records"),
            "repeated_body_policy_rows": int(keys.duplicated().sum()),
            "conflicting_rows": int(conflicts.sum()),
            "conflicting_groups": len(keys.loc[conflicts].drop_duplicates()),
            "folds": fold_audit,
            "training_sensitivities": filters,
            "actual_fitted_models": len(folds) + sum(not f["reused_primary"] for f in filters),
            "primary_fitted_models": len(folds),
            "embedding_cache": cache,
            "support_stress": support_stress(frame, vectors, plan["seed"]),
            "confirmation_labels_accessed": False,
            "candidate_promoted": False,
            "metric_note": "Policy-macro AUC is the competition-oriented primary metric; "
            "no Kaggle submission score is claimed.",
        },
    }


def execute_study(
    root: Path, directory: Path, frame: pd.DataFrame, vectors: np.ndarray, plan: dict, cache: dict
) -> None:
    """Execute only on an explicitly supplied development cohort; no source download."""
    splits = design(frame, plan)

    def save_design(path):
        atomic_json(path / "splits.json", splits)
        atomic_json(path / "row_ids.json", frame.row_id.tolist())

    stage(directory, "design", save_design)
    structural, structural_names = structural_candidates(frame)
    semantic, semantic_names = semantic_candidates(vectors)
    parts, screens, importance, selections, folds, filters = [], [], [], {}, [], []
    frozen = support_scores(vectors)
    with Progress(root / "logs/expanded.jsonl", "four_policy_feature_study") as log:
        for protocol, records in splits.items():
            for fold, assignment in enumerate(records):
                ti, vi = assignment["train"], assignment["valid"]
                training, validation = frame.iloc[ti], frame.iloc[vi]
                prefix = f"{protocol}_{fold}"
                banks = fold_bank(
                    directory,
                    prefix,
                    training,
                    validation,
                    (structural[ti], structural[vi], structural_names),
                    (semantic[ti], semantic[vi], semantic_names),
                    lambda: None,
                )
                for family, path in banks.items():
                    screens.append(
                        {
                            "protocol": protocol,
                            "fold": fold,
                            **json.loads((path / "details.json").read_text()),
                        }
                    )
                    selections[(protocol, fold, family)] = set(
                        json.loads((path / "selected.json").read_text())
                    )
                word, character = joblib.load(directory / f"{prefix}_vocabulary/vocabulary.joblib")
                matrices = _matrices(training, validation, word, character, vectors, ti, vi)
                primary = {}
                models = [*plan["lexical_references"], *VARIANTS, *plan["representation_controls"]]
                for name in models:

                    def fit(
                        path,
                        name=name,
                        training=training,
                        validation=validation,
                        banks=banks,
                        matrices=matrices,
                        protocol=protocol,
                        fold=fold,
                    ):
                        with warnings.catch_warnings():
                            warnings.simplefilter("error", ConvergenceWarning)
                            if name in VARIANTS:
                                _model(path, training, validation, banks, VARIANTS[name], fold)
                            elif name in plan["lexical_references"]:
                                _reference(path, training, validation, name)
                            else:
                                fit_lexical(
                                    path, name, matrices, training, validation, protocol, fold
                                )

                    path = stage(directory, f"{prefix}_model_{name}", fit)
                    primary[name] = path
                    part = pd.read_csv(path / "predictions.csv")
                    part["model"], part["protocol"], part["fold"] = name, protocol, fold
                    parts.append(part)
                    details = json.loads((path / "details.json").read_text())
                    folds.append(
                        {
                            "protocol": protocol,
                            "fold": fold,
                            "model": name,
                            "metrics": evaluate(part.rule_violation, part.probability, part.rule),
                            "features": details.get("retained_features", details.get("features")),
                        }
                    )
                    importance.extend(
                        {"protocol": protocol, "fold": fold, "model": name, **item}
                        for item in details.get("importance", [])
                    )
                for sensitivity in plan["sensitivity_training_filters"]:

                    def save_filter(
                        path, sensitivity=sensitivity, training=training, validation=validation
                    ):
                        retained = filter_training(training, validation, sensitivity)
                        if retained.rule_violation.nunique() != 2:
                            raise ValueError("Training sensitivity removes a class")
                        atomic_json(path / "rows.json", retained.row_id.tolist())

                    selection = stage(directory, f"{prefix}_filter_{sensitivity}", save_filter)
                    ids = json.loads((selection / "rows.json").read_text())
                    retained = (
                        training.set_index("row_id", drop=False).loc[ids].reset_index(drop=True)
                    )
                    for name in plan["sensitivity_refits"]:

                        def refit(
                            path,
                            retained=retained,
                            validation=validation,
                            name=name,
                            primary=primary,
                            protocol=protocol,
                            fold=fold,
                        ):
                            with warnings.catch_warnings():
                                warnings.simplefilter("error", ConvergenceWarning)
                                _sensitivity_fit(
                                    path, retained, validation, name, primary[name], protocol, fold
                                )

                        path = stage(directory, f"{prefix}_{sensitivity}_{name}", refit)
                        part = pd.read_csv(path / "predictions.csv")
                        part["model"], part["protocol"], part["fold"] = (
                            f"{name}_{sensitivity}",
                            protocol,
                            fold,
                        )
                        parts.append(part)
                        filters.append(
                            {
                                "protocol": protocol,
                                "fold": fold,
                                "model": name,
                                "sensitivity": sensitivity,
                                "before": len(training),
                                "after": len(retained),
                                "reused_primary": (path / "reuse.json").exists(),
                            }
                        )
                for name, values in frozen.items():
                    part = validation[["row_id", "rule", "rule_violation"]].copy()
                    part["probability"] = values[vi]
                    part["model"], part["protocol"], part["fold"] = name, protocol, fold
                    parts.append(part)
                log.emit("fold_completed", protocol=protocol, fold=fold, primary_models=len(models))
        oof = pd.concat(parts, ignore_index=True)
        validate_oof(frame, oof, plan, splits)

        def review(path):
            report = summarize(
                frame,
                oof,
                splits,
                screens,
                importance,
                selections,
                folds,
                filters,
                vectors,
                plan,
                cache,
            )
            for name, value in report.items():
                atomic_json(path / name, value)
            atomic_bytes(path / "oof.csv", oof.to_csv(index=False).encode())

        stage(directory, "review", review)
        log.emit("study_completed", run_id=directory.name, fitted_models=len(folds))


def implementation(root: Path) -> dict:
    return {name: digest(root / "src/jigsaw_rules" / name) for name in SOURCES}


def run_expanded(root: Path, *, encode: bool = False) -> Path:
    root = root.resolve()
    plan, frame = load_plan(root), load_development(root)
    if encode:
        extend_embeddings(
            root, frame, workers=plan["embedding_workers"], shard_size=plan["embedding_shard_size"]
        )
    vectors, cache = cached_vectors(root, frame)
    boundary = released_evidence(root)
    provenance = {
        "schema": 1,
        "data_kind": "post_competition_research",
        "protocol_commit": PROTOCOL_COMMIT,
        "plan_sha256": digest(root / "configs/expanded.json"),
        "sources": implementation(root),
        "boundary_run": boundary["metadata"]["run_id"],
        "boundary_checkpoint_sha256": boundary["metadata"]["private_checkpoint_sha256"],
        "original_training_sha256": digest(root / "data/raw/train.csv"),
        "development_sha256": content_key(frame.to_csv(index=False)),
        "embedding_cache": cache,
        "environment": environment(),
    }
    directory = root / "runs/expanded" / content_key(provenance)[:20]
    with FileLock(str(root / "runs/expanded.lock"), timeout=1):
        atomic_json(directory / "provenance.json", provenance)
        execute_study(root, directory, frame, vectors, plan, cache)
        export_expanded(root, directory)
    return directory


def export_expanded(root: Path, directory: Path) -> None:
    provenance = json.loads((directory / "provenance.json").read_text())
    if (
        provenance["sources"] != implementation(root)
        or directory.name != content_key(provenance)[:20]
    ):
        raise ValueError("Expanded study source or identity differs")
    frame = load_development(root)
    if provenance["development_sha256"] != content_key(frame.to_csv(index=False)):
        raise ValueError("Expanded study development source differs")
    review = directory / "review"
    marker = json.loads((review / "complete.json").read_text())
    for name in [*PUBLIC_FILES, "oof.csv"]:
        if marker.get("files", {}).get(name) != digest(review / name):
            raise ValueError("Expanded review checkpoint checksum mismatch")
    oof = pd.read_csv(review / "oof.csv")
    validate_oof(frame, oof, load_plan(root), design(frame, load_plan(root)))
    for result in json.loads((review / "results.json").read_text()):
        p = aligned_predictions(
            frame, oof[(oof.protocol == result["protocol"]) & (oof.model == result["model"])]
        )
        actual = evaluate(frame.rule_violation, p, frame.rule)
        for metric in ("rule_macro_auc", "pooled_auc", "log_loss", "brier"):
            if abs(actual[metric] - result["metrics"][metric]) > 1e-12:
                raise ValueError("Published metric differs from saved predictions")
    public = root / "reports/expanded"
    for name in PUBLIC_FILES:
        atomic_bytes(public / name, (review / name).read_bytes())
    atomic_json(
        public / "metadata.json",
        {
            **provenance,
            "run_id": directory.name,
            "files": {name: digest(review / name) for name in PUBLIC_FILES},
            "private_checkpoint_sha256": digest(review / "complete.json"),
        },
    )


def expanded_evidence(root: Path) -> dict | None:
    path = root / "reports/expanded/metadata.json"
    if not path.exists():
        return None
    metadata = json.loads(path.read_text())
    boundary = released_evidence(root)
    if (
        metadata.get("sources") != implementation(root)
        or metadata.get("plan_sha256") != digest(root / "configs/expanded.json")
        or metadata.get("protocol_commit") != PROTOCOL_COMMIT
        or metadata.get("data_kind") != "post_competition_research"
        or boundary is None
        or metadata.get("boundary_checkpoint_sha256")
        != boundary["metadata"]["private_checkpoint_sha256"]
        or set(metadata.get("files", {})) != set(PUBLIC_FILES)
    ):
        raise ValueError("Stale expanded feature evidence")
    identity = {
        k: v
        for k, v in metadata.items()
        if k not in {"run_id", "files", "private_checkpoint_sha256"}
    }
    if metadata.get("run_id") != content_key(identity)[:20]:
        raise ValueError("Expanded evidence run identity mismatch")
    result = {"metadata": metadata}
    for name, sha in metadata["files"].items():
        if digest(path.parent / name) != sha:
            raise ValueError("Expanded evidence checksum mismatch")
        result[name.removesuffix(".json")] = json.loads((path.parent / name).read_text())
    return result
