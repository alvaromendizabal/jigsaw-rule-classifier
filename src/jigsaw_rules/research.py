"""Resumable feature research on unchanged, purged competition splits.

No encoder inference, holdout-driven feature screening, model promotion or
submission generation occurs in this experiment.
"""

from __future__ import annotations

import json
import time
import warnings
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from filelock import FileLock
from scipy import sparse
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import normalize as unit_normalize

from jigsaw_rules.context import ContextEncoder, ReferenceRanks
from jigsaw_rules.data import EXAMPLES, load_data, normalize
from jigsaw_rules.embeddings import QwenEncoder, content_key, load_spec
from jigsaw_rules.metrics import evaluate
from jigsaw_rules.representations import (
    TrainingScreen,
    lexical_candidates,
    semantic_candidates,
    structural_candidates,
)
from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest, fingerprint, stage
from jigsaw_rules.semantic import input_texts, reference_splits
from jigsaw_rules.uncertainty import paired_auc_comparisons

BUDGETS = {
    "word": 4096,
    "character": 4096,
    "structure": 128,
    "lexical": 64,
    "support_tokens": 1024,
    "semantic": 256,
    "semantic_scalar": 48,
    "ranks": 64,
    "community": 128,
    "target_context": 9,
}
CORE = ("word", "character", "structure", "lexical", "support_tokens", "semantic_scalar", "ranks")
VARIANTS = {
    "word_screened": ("word",),
    **{f"word_{family}": ("word", family) for family in BUDGETS if family != "word"},
    "semantic_scalar_only": ("semantic_scalar",),
    "community_only": ("community",),
    "all_transfer": CORE,
    "all_with_coordinates": (*CORE, "semantic"),
    "all_with_metadata": (*CORE, "community", "target_context"),
    **{
        f"without_{family}": tuple(name for name in CORE if name != family)
        for family in CORE
        if family != "word"
    },
}
PUBLIC_FILES = (
    "results.json",
    "uncertainty.json",
    "screening.json",
    "importance.json",
    "catalog.json",
)


def implementation_hash(root: Path) -> str:
    """Bind public results to modeling and validation, not notebook rendering."""
    names = (
        "research.py",
        "representations.py",
        "context.py",
        "data.py",
        "embeddings.py",
        "semantic.py",
        "metrics.py",
        "runtime.py",
        "uncertainty.py",
    )
    return content_key({name: digest(root / "src/jigsaw_rules" / name) for name in names})


def read_embedding_cache(cache: Path, contract: dict, keys: list[str]) -> tuple[np.ndarray, dict]:
    """Read verified individual input hashes across shards without encoder inference."""
    if not keys or json.loads((cache / "contract.json").read_text()) != contract:
        raise ValueError("Missing or incompatible frozen embedding contract")
    required = set(keys)
    found, shards = {}, {}
    widths = set()
    for directory in sorted(cache.glob("batch_*")):
        if not directory.is_dir():
            continue
        if directory.is_symlink():
            raise ValueError("Embedding cache must not contain symlinked shards")
        marker = json.loads((directory / "complete.json").read_text())
        for name in ("inputs.json", "vectors.npy", "statistics.json"):
            if marker.get("files", {}).get(name) != digest(directory / name):
                raise ValueError(f"Embedding shard checksum mismatch: {directory.name}/{name}")
        hashes = json.loads((directory / "inputs.json").read_text())
        vectors = np.load(directory / "vectors.npy", allow_pickle=False)
        if (
            vectors.ndim != 2
            or len(hashes) != len(vectors)
            or len(hashes) != len(set(hashes))
            or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)
        ):
            raise ValueError("Invalid frozen embedding shard schema")
        widths.add(vectors.shape[1])
        if len(widths) != 1:
            raise ValueError("Embedding dimensionality differs across shards")
        shards[directory.name] = digest(directory / "complete.json")
        for key, vector in zip(hashes, vectors, strict=True):
            if key in required:
                if key in found and not np.array_equal(found[key], vector):
                    raise ValueError("Conflicting cached vectors for the same input")
                found[key] = vector
    if required - found.keys():
        raise ValueError(
            f"Frozen cache lacks {len(required - found.keys())} required inputs; "
            "no encoding was started"
        )
    return np.stack([found[key] for key in keys]), {
        "cache": cache.name,
        "contract_sha256": content_key(contract),
        "shards_sha256": content_key(shards),
        "shards_verified": len(shards),
        "requested_inputs": len(keys),
        "unique_requested_inputs": len(required),
        "newly_encoded_inputs": 0,
    }


def cached_vectors(root: Path, frame: pd.DataFrame) -> tuple[np.ndarray, dict]:
    spec = load_spec(root)
    contract = QwenEncoder(root, spec).contract
    keys = [content_key(text) for text in input_texts(frame, spec)]
    cache = root / "runs/embeddings" / content_key(contract)[:20]
    array, details = read_embedding_cache(cache, contract, keys)
    return array.reshape(len(frame), 5, -1), details


def _save_representation(directory: Path, array: np.ndarray, names: list[str]) -> None:
    if array.ndim != 2 or array.shape[1] != len(names) or not np.isfinite(array).all():
        raise ValueError("Invalid candidate representation")
    np.savez_compressed(directory / "matrix.npz", values=array)
    atomic_json(directory / "names.json", names)


def _vocabulary(directory: Path, training: pd.DataFrame) -> None:
    corpus = [text for column in ("body", "rule", *EXAMPLES) for text in training[column]]
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=60000)
    character = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=60000
    )
    word.fit(corpus)
    character.fit(corpus)
    joblib.dump((word, character), directory / "vocabulary.joblib", compress=3)
    atomic_json(
        directory / "corpus.json",
        {
            "texts": len(corpus),
            "corpus_sha256": content_key(corpus),
            "training_row_ids_sha256": content_key(training.row_id.tolist()),
            "scope": "Only supplied text from the purged training partition",
        },
    )


def _family(
    directory: Path,
    name: str,
    training: pd.DataFrame,
    validation: pd.DataFrame,
    raw_train,
    raw_valid,
    names: list[str],
) -> None:
    before = time.monotonic()
    screen = TrainingScreen(
        BUDGETS[name], correlation=None if sparse.issparse(raw_train) else 0.995
    )
    screen.fit(raw_train, training.rule_violation.to_numpy(), names)
    selected_train, selected_valid = (
        screen.transform(raw_train, names),
        screen.transform(raw_valid, names),
    )
    scaler = None
    if not sparse.issparse(selected_train) and selected_train.shape[1]:
        scaler = StandardScaler().fit(selected_train)
        scale = np.sqrt(selected_train.shape[1])
        selected_train = scaler.transform(selected_train) / scale
        selected_valid = scaler.transform(selected_valid) / scale
    elif sparse.issparse(selected_train) and selected_train.shape[1]:
        selected_train = unit_normalize(selected_train, norm="l2")
        selected_valid = unit_normalize(selected_valid, norm="l2")
    for label, matrix in (("train", selected_train), ("valid", selected_valid)):
        matrix = sparse.csr_matrix(matrix)
        if not np.isfinite(matrix.data).all():
            raise ValueError("Nonfinite screened feature matrix")
        sparse.save_npz(directory / f"{label}.npz", matrix)
    joblib.dump((screen, scaler), directory / "screen.joblib", compress=3)
    atomic_bytes(directory / "catalog.csv", screen.catalog_.to_csv(index=False).encode())
    atomic_json(directory / "selected.json", [names[index] for index in screen.indices_])
    atomic_json(
        directory / "details.json",
        {
            **screen.audit_,
            "family": name,
            "screen_seconds": time.monotonic() - before,
            "validation_rows": len(validation),
            "train_row_ids_sha256": content_key(training.row_id.tolist()),
            "validation_row_ids_sha256": content_key(validation.row_id.tolist()),
        },
    )


def fold_bank(
    directory: Path,
    prefix: str,
    training: pd.DataFrame,
    validation: pd.DataFrame,
    structural: tuple[np.ndarray, np.ndarray, list[str]],
    semantic: tuple[np.ndarray, np.ndarray, list[str]],
    checkpoint,
) -> dict:
    vocabulary = stage(directory, prefix + "_vocabulary", lambda path: _vocabulary(path, training))
    checkpoint()
    word, character = joblib.load(vocabulary / "vocabulary.joblib")
    banks = {}
    for name in BUDGETS:

        def build(path, name=name):
            if name in {"word", "character"}:
                vectorizer = word if name == "word" else character
                x_train, x_valid = (
                    vectorizer.transform(training.body),
                    vectorizer.transform(validation.body),
                )
                names = vectorizer.get_feature_names_out().tolist()
            elif name in {"structure", "lexical", "ranks"}:
                lexical_train, lexical_names = lexical_candidates(training, word, character)
                lexical_valid, _ = lexical_candidates(validation, word, character)
                if name == "structure":
                    x_train, x_valid, names = structural
                elif name == "lexical":
                    x_train, x_valid, names = lexical_train, lexical_valid, lexical_names
                else:
                    ranker = ReferenceRanks().fit(lexical_train)
                    x_train, x_valid = (
                        ranker.transform(lexical_train),
                        ranker.transform(lexical_valid),
                    )
                    names = [f"training_percentile/{item}" for item in lexical_names]
                    joblib.dump(ranker, path / "ranker.joblib", compress=3)
            elif name in {"semantic", "semantic_scalar"}:
                x_train, x_valid, names = semantic
                indices = [
                    i
                    for i, item in enumerate(names)
                    if item.startswith("similarity/") == (name == "semantic_scalar")
                ]
                x_train, x_valid = x_train[:, indices], x_valid[:, indices]
                names = [names[i] for i in indices]
            elif name == "support_tokens":

                def contrasts(frame):
                    support = [word.transform(frame[column]) for column in EXAMPLES]
                    direction = (support[0] + support[1] - support[2] - support[3]) / 2
                    return word.transform(frame.body).multiply(direction).tocsr()

                x_train, x_valid = contrasts(training), contrasts(validation)
                names = [
                    f"comment_x_support_direction/{term}" for term in word.get_feature_names_out()
                ]
            elif name == "target_context":
                encoder = ContextEncoder()
                x_train, names = encoder.fit_transform(training)
                x_valid, _ = encoder.transform(validation.drop(columns="rule_violation"))
                joblib.dump(encoder, path / "context.joblib", compress=3)
                atomic_json(path / "inner_folds.json", encoder.inner_audit_)
            else:
                encoder = OneHotEncoder(
                    handle_unknown="ignore", sparse_output=True, dtype=np.float64
                )
                x_train = encoder.fit_transform(training[["subreddit"]])
                x_valid = encoder.transform(validation[["subreddit"]])
                counts = training.subreddit.value_counts()
                frequency_train = np.log1p(training.subreddit.map(counts).to_numpy()) / np.log1p(
                    len(training)
                )
                frequency_valid = np.log1p(
                    validation.subreddit.map(counts).fillna(0).to_numpy()
                ) / np.log1p(len(training))
                x_train = sparse.hstack(
                    [x_train, sparse.csr_matrix(frequency_train[:, None])], format="csr"
                )
                x_valid = sparse.hstack(
                    [x_valid, sparse.csr_matrix(frequency_valid[:, None])], format="csr"
                )
                names = encoder.get_feature_names_out().tolist() + ["training_frequency"]
                joblib.dump((encoder, counts), path / "community.joblib", compress=3)
            _family(path, name, training, validation, x_train, x_valid, names)

        banks[name] = stage(directory, prefix + "_" + name, build)
        checkpoint()
    return banks


def _model(
    destination: Path,
    training: pd.DataFrame,
    validation: pd.DataFrame,
    banks: dict,
    families: tuple[str, ...],
    fold: int,
) -> None:
    before = time.monotonic()
    x_train = sparse.hstack(
        [sparse.load_npz(banks[name] / "train.npz") for name in families], format="csr"
    )
    blocks = {name: sparse.load_npz(banks[name] / "valid.npz") for name in families}
    x_valid = sparse.hstack(list(blocks.values()), format="csr")
    if not x_train.shape[1]:
        raise ValueError("All candidate features were rejected; model is not estimable")
    classifier = LogisticRegression(C=2.0, solver="liblinear", max_iter=2000, random_state=2025)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(x_train, training.rule_violation)
    prediction = classifier.predict_proba(x_valid)[:, 1]
    part = validation[["row_id", "rule", "rule_violation"]].copy()
    part["probability"], part["fold"] = prediction, fold
    atomic_bytes(destination / "predictions.csv", part.to_csv(index=False).encode())
    joblib.dump(classifier, destination / "model.joblib", compress=3)
    fold_metrics = evaluate(validation.rule_violation, prediction, validation.rule)
    base_auc = fold_metrics["rule_macro_auc"]
    logit = classifier.decision_function(x_valid)
    rng = np.random.default_rng(2025 + fold)
    importance, offset = [], 0
    for family, block in blocks.items():
        width = block.shape[1]
        contribution = np.asarray(block @ classifier.coef_[0, offset : offset + width]).ravel()
        drops = []
        for _ in range(3):
            shuffled = contribution.copy()
            for rule in sorted(set(validation.rule)):
                indices = np.flatnonzero(validation.rule.to_numpy() == rule)
                shuffled[indices] = contribution[rng.permutation(indices)]
            permuted = expit(logit - contribution + shuffled)
            drops.append(
                base_auc
                - evaluate(validation.rule_violation, permuted, validation.rule)["rule_macro_auc"]
            )
        importance.append(
            {
                "family": family,
                "retained_features": width,
                "mean_absolute_logit_contribution": float(np.abs(contribution).mean()),
                "within_rule_permutation_auc_drops": drops,
                "purpose": "Post-fit validation interpretation only; never used to select features",
            }
        )
        offset += width
    coefficient_names = [
        f"{family}/{name}"
        for family in families
        for name in json.loads((banks[family] / "selected.json").read_text())
    ]
    atomic_bytes(
        destination / "coefficients.csv",
        pd.DataFrame({"feature": coefficient_names, "coefficient": classifier.coef_[0]})
        .to_csv(index=False)
        .encode(),
    )
    atomic_json(
        destination / "details.json",
        {
            "seconds": time.monotonic() - before,
            "training_rows": len(training),
            "validation_rows": len(validation),
            "retained_features": x_train.shape[1],
            "solver_iterations": classifier.n_iter_.tolist(),
            "metrics": fold_metrics,
            "importance": importance,
        },
    )


def run_research(root: Path, baseline_run: str, *, cloud: dict | None = None) -> Path:
    root = Path(root).resolve()
    (root / "runs").mkdir(exist_ok=True)
    with (
        FileLock(str(root / "runs/research.lock"), timeout=1),
        Progress(root / "logs/research.jsonl", "feature_research") as log,
    ):
        train, _, _ = load_data(root / "data/raw")
        if (root / "data/raw/SYNTHETIC.txt").exists():
            raise ValueError(
                "Broad research requires verified competition data and frozen embeddings"
            )
        splits = reference_splits(root, train, baseline_run)
        vectors, cache = cached_vectors(root, train)
        reference_directory = root / "runs" / baseline_run / "review"
        marker = json.loads((reference_directory / "complete.json").read_text())
        if marker["files"].get("oof.csv") != digest(reference_directory / "oof.csv"):
            raise ValueError("Reference OOF checksum mismatch")
        reference = pd.read_csv(reference_directory / "oof.csv")
        config = {
            "experiment": "broad_feature_research",
            "baseline_run": baseline_run,
            "variants": VARIANTS,
            "budgets": BUDGETS,
            "splits": splits,
            "seed": 2025,
            "embedding_cache": cache,
            "implementation_sha256": implementation_hash(root),
            "reference_oof_sha256": digest(reference_directory / "oof.csv"),
        }
        _, provenance = fingerprint(root / "data/raw", config)
        # Unrelated CLI/reporting additions must not invalidate completed research.
        provenance["code"] = {
            name: provenance["code"][name]
            for name in (
                "research.py",
                "representations.py",
                "context.py",
                "data.py",
                "embeddings.py",
                "semantic.py",
                "metrics.py",
                "runtime.py",
                "uncertainty.py",
            )
        }
        run_id = content_key(provenance)[:20]
        directory = root / "runs" / run_id
        atomic_json(directory / "provenance.json", provenance)

        def checkpoint():
            if cloud:
                from jigsaw_rules.cloud import backup

                backup(root, cloud["bucket"], cloud["region"])

        log.emit("RESEARCH_IDENTIFIED", run_id=run_id, cache=cache, variants=list(VARIANTS))
        raw_structure = stage(
            directory,
            "structural_representation",
            lambda path: _save_representation(path, *structural_candidates(train)),
        )
        checkpoint()
        raw_semantic = stage(
            directory,
            "semantic_representation",
            lambda path, vectors=vectors: _save_representation(path, *semantic_candidates(vectors)),
        )
        checkpoint()
        structure_array = np.load(raw_structure / "matrix.npz", allow_pickle=False)["values"]
        structure_names = json.loads((raw_structure / "names.json").read_text())
        semantic_array = np.load(raw_semantic / "matrix.npz", allow_pickle=False)["values"]
        semantic_names = json.loads((raw_semantic / "names.json").read_text())
        del vectors
        oof_parts, screens, interpretation, selections, times = [], [], [], {}, {}
        fold_metrics = []
        for protocol, assignments in splits.items():
            for fold, assignment in enumerate(assignments):
                ti, vi = assignment["train"], assignment["valid"]
                training, validation = train.iloc[ti], train.iloc[vi]
                prefix = f"{protocol}_{fold}"
                banks = fold_bank(
                    directory,
                    prefix,
                    training,
                    validation,
                    (structure_array[ti], structure_array[vi], structure_names),
                    (semantic_array[ti], semantic_array[vi], semantic_names),
                    checkpoint,
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
                for variant, families in VARIANTS.items():

                    def fit_variant(
                        path,
                        families=families,
                        training=training,
                        validation=validation,
                        banks=banks,
                        fold=fold,
                    ):
                        _model(path, training, validation, banks, families, fold)

                    completed = stage(directory, f"{prefix}_model_{variant}", fit_variant)
                    part = pd.read_csv(completed / "predictions.csv")
                    part["model"], part["protocol"] = variant, protocol
                    oof_parts.append(part)
                    details = json.loads((completed / "details.json").read_text())
                    fold_metrics.append(
                        {
                            "protocol": protocol,
                            "model": variant,
                            "fold": fold,
                            "retained_features": details["retained_features"],
                            "metrics": details["metrics"],
                        }
                    )
                    times[(protocol, variant)] = (
                        times.get((protocol, variant), 0) + details["seconds"]
                    )
                    interpretation.extend(
                        {"protocol": protocol, "fold": fold, "model": variant, **item}
                        for item in details["importance"]
                    )
                    checkpoint()
                log.emit(
                    "RESEARCH_FOLD_COMPLETED", protocol=protocol, fold=fold, models=len(VARIANTS)
                )
        combined = pd.concat(oof_parts, ignore_index=True)
        word_for_catalog, character_for_catalog = joblib.load(
            directory / "seen_rule_0_vocabulary/vocabulary.joblib"
        )

        def review(path):
            results, uncertainty, stability = [], [], []
            for protocol in splits:
                ref = (
                    reference[
                        (reference.protocol == protocol) & (reference.model == "rule_examples")
                    ]
                    .set_index("row_id")
                    .loc[train.row_id]
                )
                if (
                    len(ref) != len(train)
                    or not np.array_equal(ref.rule_violation, train.rule_violation)
                    or not np.array_equal(ref.rule, train.rule)
                ):
                    raise ValueError("Reference OOF labels or rules differ")
                for variant in VARIANTS:
                    rows = combined[(combined.protocol == protocol) & (combined.model == variant)]
                    if rows.row_id.duplicated().any() or set(rows.row_id) != set(train.row_id):
                        raise ValueError("Expected exactly one prediction per row and protocol")
                    rows = rows.set_index("row_id").loc[train.row_id]
                    results.append(
                        {
                            "model": variant,
                            "protocol": protocol,
                            "run_id": run_id,
                            "metrics": evaluate(rows.rule_violation, rows.probability, rows.rule),
                            "training_rows": len(train),
                            "data_kind": "competition",
                            "fit_and_interpret_seconds": times[(protocol, variant)],
                        }
                    )
                prediction_bank = {
                    variant: combined[(combined.protocol == protocol) & (combined.model == variant)]
                    .set_index("row_id")
                    .loc[train.row_id]
                    .probability.to_numpy()
                    for variant in VARIANTS
                }
                prediction_bank["original_reference"] = ref.probability.to_numpy()
                contrasts = [(variant, variant, "original_reference") for variant in VARIANTS]
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
                    {"protocol": protocol, "model": item["candidate"], **item}
                    for item in paired_auc_comparisons(
                        train.rule_violation,
                        prediction_bank,
                        train.rule,
                        train.body.map(normalize),
                        contrasts,
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
            atomic_json(path / "results.json", results)
            atomic_json(path / "uncertainty.json", uncertainty)
            atomic_json(
                path / "screening.json",
                {"folds": screens, "stability": stability, "model_folds": fold_metrics},
            )
            atomic_json(path / "importance.json", interpretation)
            atomic_json(
                path / "catalog.json",
                {
                    "variants": VARIANTS,
                    "budgets": BUDGETS,
                    "embedding_cache": cache,
                    "structural_candidate_names": structure_names,
                    "semantic_candidate_groups": [
                        "comment",
                        "support_direction",
                        "comment_x_direction",
                        "relative_absolute_distance",
                        "positive_spread",
                        "negative_spread",
                        "similarity",
                    ],
                    "structural_candidates": len(structure_names),
                    "semantic_candidates": len(semantic_names),
                    "lexical_interaction_candidates": len(
                        lexical_candidates(train.iloc[:1], word_for_catalog, character_for_catalog)[
                            1
                        ]
                    ),
                    "notes": [
                        "Word and character vocabularies are fitted separately inside every "
                        "purged training fold.",
                        "Vocabulary terms, row-level predictions and full candidate "
                        "decisions remain private checkpoints.",
                        "Only two labeled rules are available; intervals condition on those "
                        "rules and fixed predictions.",
                        "Predeclared variants are exploratory; intervals are conditional and "
                        "not multiplicity-adjusted.",
                        "Community features are a shortcut diagnostic, not an automatically "
                        "selected deployment feature.",
                        "No timestamps justify temporal features. Metadata targets cross-fit "
                        "on three inner purged comment groups; no external labels were "
                        "added.",
                        "No model is automatically promoted, tuned, calibrated or submitted "
                        "by this experiment.",
                    ],
                },
            )
            atomic_bytes(path / "oof.csv", combined.to_csv(index=False).encode())

        stage(directory, "review", review)
        atomic_json(directory / "status.json", {"status": "completed", "synthetic": False})
        checkpoint()
        log.emit("RESEARCH_COMPLETED", run_id=run_id, models=len(VARIANTS), protocols=len(splits))
        return directory


def export_research(root: Path, directory: Path) -> None:
    from jigsaw_rules.review import public_evidence

    verify_research(root, directory)
    provenance = json.loads((directory / "provenance.json").read_text())
    if json.loads((directory / "status.json").read_text()).get("status") != "completed":
        raise ValueError("Cannot export incomplete research")
    baseline = public_evidence(root, "baseline")
    if (
        provenance["data"]["train.csv"] != baseline["training_sha256"]
        or provenance["config"]["baseline_run"] != baseline["run_id"]
        or provenance["config"]["implementation_sha256"] != implementation_hash(root)
    ):
        raise ValueError("Research provenance differs from current source or reference data")
    marker = json.loads((directory / "review/complete.json").read_text())
    for name in PUBLIC_FILES:
        if marker["files"].get(name) != digest(directory / "review" / name):
            raise ValueError("Research aggregate checksum mismatch")
    destination = root / "reports/research"
    for name in PUBLIC_FILES:
        atomic_bytes(destination / name, (directory / "review" / name).read_bytes())
    atomic_json(
        destination / "metadata.json",
        {
            "schema": 1,
            "data_kind": "competition",
            "run_id": directory.name,
            "baseline_run": baseline["run_id"],
            "training_sha256": baseline["training_sha256"],
            "source_sha256": provenance["code"],
            "implementation_sha256": implementation_hash(root),
            "files": {name: digest(destination / name) for name in PUBLIC_FILES},
            "selection": "Exploratory feature research; no automatic model promotion or "
            "leaderboard claim.",
        },
    )


def research_evidence(root: Path) -> dict | None:
    from jigsaw_rules.review import public_evidence

    path = root / "reports/research/metadata.json"
    if not path.exists():
        return None
    metadata = json.loads(path.read_text())
    baseline = public_evidence(root, "baseline")
    if (
        metadata.get("schema") != 1
        or metadata.get("data_kind") != "competition"
        or set(metadata.get("files", {})) != set(PUBLIC_FILES)
        or metadata.get("implementation_sha256") != implementation_hash(root)
        or metadata.get("training_sha256") != baseline["training_sha256"]
        or metadata.get("baseline_run") != baseline["run_id"]
    ):
        raise ValueError("Stale, synthetic or incompatible public research evidence")
    result = {"metadata": metadata}
    for name, sha in metadata["files"].items():
        if digest(path.parent / name) != sha:
            raise ValueError("Research aggregate checksum mismatch")
        result[name.removesuffix(".json")] = json.loads((path.parent / name).read_text())
    expected = {
        (protocol, model) for protocol in ("seen_rule", "heldout_rule") for model in VARIANTS
    }
    actual = {(row["protocol"], row["model"]) for row in result["results"]}
    if actual != expected or len(result["results"]) != len(expected):
        raise ValueError("Research results have missing or duplicate variants")
    for record in result["screening"]["folds"]:
        if (
            record["candidates"] != record["retained"] + record["rejected"]
            or sum(record["decisions"].values()) != record["candidates"]
        ):
            raise ValueError("Research screening counts do not reconcile")
    return result


def verify_research(root: Path, directory: Path) -> dict:
    """Recompute every OOF metric and verify all completed artifacts, without fitting."""
    train, _, _ = load_data(root / "data/raw")
    provenance = json.loads((directory / "provenance.json").read_text())
    if content_key(provenance)[:20] != directory.name:
        raise ValueError("Research run identity differs from its provenance")
    if provenance["data"]["train.csv"] != digest(root / "data/raw/train.csv"):
        raise ValueError("Research data lineage differs")
    verified = 0
    for marker in directory.glob("*/complete.json"):
        for name, sha in json.loads(marker.read_text())["files"].items():
            path = marker.parent / name
            if not path.resolve().is_relative_to(directory.resolve()) or digest(path) != sha:
                raise ValueError("Research checkpoint checksum mismatch")
            verified += 1
    saved = pd.read_csv(directory / "review/oof.csv")
    records = json.loads((directory / "review/results.json").read_text())
    expected = {
        (protocol, model) for protocol in ("seen_rule", "heldout_rule") for model in VARIANTS
    }
    if {(r["protocol"], r["model"]) for r in records} != expected or len(records) != len(expected):
        raise ValueError("Research result coverage differs")
    for record in records:
        part = saved[(saved.protocol == record["protocol"]) & (saved.model == record["model"])]
        if part.row_id.duplicated().any() or set(part.row_id) != set(train.row_id):
            raise ValueError("Research OOF row coverage differs")
        part = part.set_index("row_id").loc[train.row_id]
        if not np.array_equal(part.rule_violation, train.rule_violation):
            raise ValueError("Research OOF labels differ")
        if not np.array_equal(part.rule, train.rule):
            raise ValueError("Research OOF rules differ")
        metrics = evaluate(part.rule_violation, part.probability, part.rule)
        for metric in ("rule_macro_auc", "pooled_auc", "log_loss", "brier"):
            if not np.isclose(metrics[metric], record["metrics"][metric], atol=1e-12, rtol=0):
                raise ValueError("Research OOF metric recomputation differs")
    return {
        "run_id": directory.name,
        "files_verified": verified,
        "oof_metric_records_recomputed": len(records),
        "new_model_fits": 0,
    }
