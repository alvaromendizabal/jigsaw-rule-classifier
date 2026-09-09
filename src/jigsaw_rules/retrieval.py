"""Policy-excluded semantic neighborhoods: cross-fitted, target-derived features."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from filelock import FileLock
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from jigsaw_rules.data import EXAMPLES, normalize
from jigsaw_rules.embeddings import content_key
from jigsaw_rules.expanded import aligned_predictions, design, expanded_evidence, load_development
from jigsaw_rules.expanded import load_plan as development_plan
from jigsaw_rules.metrics import evaluate
from jigsaw_rules.representations import TrainingScreen
from jigsaw_rules.research import BUDGETS, _model, cached_vectors
from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest, environment, stage
from jigsaw_rules.uncertainty import paired_auc_comparisons

PROTOCOL_COMMIT = "087cec0aa08d77f40abc2f498a99731fd3d6a3db"
PLAN_SHA256 = "78633ce94cfc51c24310faa9425c98ea9fee27e582e5bbf4495c5bf259e1735d"
PUBLIC_FILES = ("results.json", "uncertainty.json", "screening.json", "importance.json")


def load_plan(root: Path) -> dict:
    path = root / "configs/retrieval.json"
    if digest(path) != PLAN_SHA256:
        raise ValueError("Retrieval plan differs from its pre-score protocol")
    return json.loads(path.read_text())


def unit(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values, axis=1, keepdims=True).clip(1e-12)


def geometry(vectors: np.ndarray) -> dict[str, np.ndarray]:
    if vectors.ndim != 3 or vectors.shape[1] != 5 or not np.isfinite(vectors).all():
        raise ValueError("Expected finite body and support vectors")
    if not np.allclose(np.linalg.norm(vectors, axis=2), 1, atol=1e-4):
        raise ValueError("Frozen body/support vectors must have unit norm")
    direction = vectors[:, 1:3].mean(1) - vectors[:, 3:5].mean(1)
    return {
        "body": unit(vectors[:, 0]),
        "support_direction": unit(direction),
        "body_x_direction": unit(vectors[:, 0] * direction),
    }


def forbidden_overlap(reference: pd.DataFrame, query: pd.DataFrame) -> np.ndarray:
    forbidden = set(query.body.map(normalize))
    return (
        reference[["body", *EXAMPLES]]
        .apply(lambda column: column.map(normalize).isin(forbidden))
        .any(axis=1)
        .to_numpy()
    )


class PolicyRetrieval:
    """References from the query policy are excluded even for familiar policies."""

    def __init__(self, plan: dict):
        self.plan = plan

    def fit(self, frame: pd.DataFrame, vectors: np.ndarray) -> PolicyRetrieval:
        if len(frame) != len(vectors) or not frame.rule_violation.isin([0, 1]).all():
            raise ValueError("Retrieval requires aligned binary training labels")
        if frame.row_id.duplicated().any():
            raise ValueError("Retrieval reference IDs must be unique")
        order = np.argsort(frame.row_id.to_numpy(), kind="stable")
        self.frame_ = frame.iloc[order].reset_index(drop=True).copy()
        self.spaces_ = geometry(np.asarray(vectors[order], dtype=np.float64))
        self.width_ = vectors.shape[2]
        return self

    def _features(self, query: pd.DataFrame, vectors: np.ndarray, reference: np.ndarray):
        if "rule_violation" in query:
            raise ValueError("Retrieval transform must not accept query targets")
        if vectors.ndim != 3 or len(query) != len(vectors) or vectors.shape[2] != self.width_:
            raise ValueError("Query vector schema or row order differs")
        ref = self.frame_.iloc[reference]
        if set(ref.rule) & set(query.rule):
            raise ValueError("Query policy entered the labeled retrieval bank")
        if forbidden_overlap(ref, query).any():
            raise ValueError("Query body leaks through a retrieval body or support")
        labels = ref.rule_violation.to_numpy()
        if set(labels.tolist()) != {0, 1}:
            raise ValueError("Retrieval bank needs both training classes")
        spaces = geometry(np.asarray(vectors, dtype=np.float64))
        values, names = [], []
        for space in self.plan["spaces"]:
            matrix, queries = self.spaces_[space][reference], spaces[space]
            positive, negative = matrix[labels == 1].mean(0), matrix[labels == 0].mean(0)
            posnorm, negnorm = np.linalg.norm(positive), np.linalg.norm(negative)
            pcos = queries @ (positive / max(1e-12, posnorm))
            ncos = queries @ (negative / max(1e-12, negnorm))
            prior = float(labels.mean())
            prototypes = np.column_stack(
                [
                    pcos,
                    ncos,
                    pcos - ncos,
                    np.full(len(query), posnorm),
                    np.full(len(query), negnorm),
                    np.full(len(query), np.linalg.norm(positive - negative)),
                    np.full(len(query), prior),
                ]
            )
            values.append(prototypes)
            names.extend(
                f"{space}/prototype/{name}"
                for name in [
                    "positive_cosine",
                    "negative_cosine",
                    "class_margin",
                    "positive_coherence",
                    "negative_coherence",
                    "separation",
                    "prior",
                ]
            )
            neighborhood = {k: [] for k in self.plan["neighbors"]}
            maximum = min(max(neighborhood), len(reference))
            for start in range(0, len(query), self.plan["batch_size"]):
                cosine = (queries[start : start + self.plan["batch_size"]] @ matrix.T).clip(-1, 1)
                selected = np.argpartition(-cosine, maximum - 1, axis=1)[:, :maximum]
                similarity = np.take_along_axis(cosine, selected, axis=1)
                order = np.argsort(-similarity, axis=1, kind="stable")
                similarity = np.take_along_axis(similarity, order, axis=1)
                selected = np.take_along_axis(selected, order, axis=1)
                for k in neighborhood:
                    sim, y = similarity[:, :k], labels[selected[:, :k]]
                    weight = np.exp((sim - 1) / self.plan["similarity_temperature"])
                    pmax = np.where(y == 1, sim, -1).max(1)
                    nmax = np.where(y == 0, sim, -1).max(1)
                    neighborhood[k].append(
                        np.column_stack(
                            [
                                y.mean(1),
                                (weight * y).sum(1) / weight.sum(1),
                                sim.mean(1),
                                sim.std(1),
                                pmax,
                                nmax,
                                pmax - nmax,
                                y.mean(1) - prior,
                                (y == 1).any(1),
                                (y == 0).any(1),
                            ]
                        )
                    )
            for k, chunks in neighborhood.items():
                values.append(np.concatenate(chunks))
                names.extend(
                    f"{space}/k{k}/{name}"
                    for name in [
                        "positive_share",
                        "weighted_positive_share",
                        "mean_cosine",
                        "cosine_spread",
                        "nearest_positive",
                        "nearest_negative",
                        "class_margin",
                        "share_minus_prior",
                        "positive_available",
                        "negative_available",
                    ]
                )
        raw = np.column_stack(values)
        continuous = [i for i, name in enumerate(names) if not name.endswith("_available")]
        scalar = raw[:, continuous]
        blocks = [raw, np.sign(scalar) * np.sqrt(np.abs(scalar)), np.sign(scalar) * scalar**2]
        all_names = (
            names
            + [f"sqrt/{names[i]}" for i in continuous]
            + [f"square/{names[i]}" for i in continuous]
        )
        body = vectors[:, 0]
        own_positive, own_negative = unit(vectors[:, 1:3].mean(1)), unit(vectors[:, 3:5].mean(1))
        own_margin = np.einsum("nd,nd->n", body, own_positive - own_negative)
        margins = [i for i, name in enumerate(names) if name.endswith("/class_margin")]
        blocks.append(raw[:, margins] * own_margin[:, None])
        all_names.extend(f"supplied_margin_x/{names[i]}" for i in margins)
        output = np.column_stack(blocks)
        if not np.isfinite(output).all():
            raise ValueError("Nonfinite retrieval feature")
        return output, all_names

    def transform(self, query: pd.DataFrame, vectors: np.ndarray):
        if "rule_violation" in query:
            raise ValueError("Retrieval transform must not accept query targets")
        if not hasattr(self, "frame_"):
            raise ValueError("Fit the retrieval reference before transformation")
        output, names = None, None
        for rule in sorted(query.rule.unique()):
            indices = np.flatnonzero(query.rule.to_numpy() == rule)
            reference = np.flatnonzero(self.frame_.rule.to_numpy() != rule)
            values, names = self._features(query.iloc[indices], vectors[indices], reference)
            if output is None:
                output = np.empty((len(query), values.shape[1]))
            output[indices] = values
        if output is None:
            raise ValueError("Nonempty retrieval query required")
        return output, names

    def fit_transform(self, frame: pd.DataFrame, vectors: np.ndarray):
        self.fit(frame, vectors)
        output, names, audit = None, None, []
        for rule in sorted(frame.rule.unique()):
            indices = np.flatnonzero(frame.rule.to_numpy() == rule)
            query = frame.iloc[indices].drop(columns="rule_violation")
            initial = np.flatnonzero(self.frame_.rule.to_numpy() != rule)
            reference = initial[~forbidden_overlap(self.frame_.iloc[initial], query)]
            values, names = self._features(query, vectors[indices], reference)
            if output is None:
                output = np.empty((len(frame), values.shape[1]))
            output[indices] = values
            audit.append(
                {
                    "query_policy": rule,
                    "query_row_ids": query.row_id.tolist(),
                    "reference_row_ids": self.frame_.iloc[reference].row_id.tolist(),
                    "reference_policies": sorted(self.frame_.iloc[reference].rule.unique()),
                    "purged_rows": len(initial) - len(reference),
                }
            )
        if output is None:
            raise ValueError("Nonempty retrieval training data required")
        self.inner_audit_ = audit
        return output, names


def build_bank(path: Path, training, validation, train_vectors, valid_vectors, plan) -> None:
    encoder = PolicyRetrieval(plan)
    raw, names = encoder.fit_transform(training, train_vectors)
    valid, valid_names = encoder.transform(validation.drop(columns="rule_violation"), valid_vectors)
    if names != valid_names:
        raise ValueError("Retrieval feature schema differs across partitions")
    screen = TrainingScreen(plan["maximum_features"], plan["correlation_threshold"])
    screen.fit(raw, training.rule_violation.to_numpy(), names)
    x, v = screen.transform(raw, names), screen.transform(valid, names)
    scaler = StandardScaler().fit(x)
    x, v = scaler.transform(x) / np.sqrt(x.shape[1]), scaler.transform(v) / np.sqrt(v.shape[1])
    sparse.save_npz(path / "train.npz", sparse.csr_matrix(x))
    sparse.save_npz(path / "valid.npz", sparse.csr_matrix(v))
    joblib.dump((encoder, screen, scaler), path / "transform.joblib", compress=3)
    atomic_json(path / "selected.json", [names[i] for i in screen.indices_])
    atomic_json(path / "inner_folds.json", encoder.inner_audit_)
    atomic_json(
        path / "details.json",
        {
            **screen.audit_,
            "train_row_ids_sha256": content_key(training.row_id.tolist()),
            "valid_row_ids_sha256": content_key(validation.row_id.tolist()),
        },
    )
    atomic_bytes(path / "catalog.csv", screen.catalog_.to_csv(index=False).encode())


def run_retrieval(root: Path) -> Path:
    root = root.resolve()
    base, plan = expanded_evidence(root), load_plan(root)
    if base is None:
        raise ValueError("Retrieval study requires completed expanded evidence")
    frame = load_development(root)
    vectors, cache = cached_vectors(root, frame)
    original = root / "runs/expanded" / base["metadata"]["run_id"]
    if digest(original / "review/complete.json") != base["metadata"]["private_checkpoint_sha256"]:
        raise ValueError("Expanded reference checkpoint differs")
    marker = json.loads((original / "review/complete.json").read_text())["files"]
    if marker["oof.csv"] != digest(original / "review/oof.csv"):
        raise ValueError("Expanded reference predictions differ")
    baseline = pd.read_csv(original / "review/oof.csv")
    identity = {
        "schema": 1,
        "data_kind": "post_competition_research",
        "protocol_commit": PROTOCOL_COMMIT,
        "plan_sha256": digest(root / "configs/retrieval.json"),
        "source_sha256": digest(Path(__file__)),
        "expanded_metadata_sha256": digest(root / "reports/expanded/metadata.json"),
        "embedding_cache": cache,
        "environment": environment(),
    }
    directory = root / "runs/retrieval" / content_key(identity)[:20]
    splits = design(frame, development_plan(root))
    parts, screens, importance = [], [], []
    with (
        FileLock(str(root / "runs/retrieval.lock"), timeout=1),
        Progress(root / "logs/retrieval.jsonl", "semantic_retrieval") as log,
    ):
        atomic_json(directory / "provenance.json", identity)
        for protocol, assignments in splits.items():
            for fold, a in enumerate(assignments):
                ti, vi = a["train"], a["valid"]
                training, validation = frame.iloc[ti], frame.iloc[vi]
                prefix = f"{protocol}_{fold}"

                def features(path, training=training, validation=validation, ti=ti, vi=vi):
                    build_bank(path, training, validation, vectors[ti], vectors[vi], plan)

                bank = stage(directory, prefix + "_retrieval", features)
                screens.append(
                    {
                        "protocol": protocol,
                        "fold": fold,
                        **json.loads((bank / "details.json").read_text()),
                    }
                )
                banks = {name: original / f"{prefix}_{name}" for name in BUDGETS}
                for path in banks.values():
                    record = json.loads((path / "complete.json").read_text())["files"]
                    for filename in ("train.npz", "valid.npz", "selected.json", "details.json"):
                        if record[filename] != digest(path / filename):
                            raise ValueError("Expanded feature bank checksum differs")
                    details = json.loads((path / "details.json").read_text())
                    if details["train_row_ids_sha256"] != content_key(
                        training.row_id.tolist()
                    ) or details["validation_row_ids_sha256"] != content_key(
                        validation.row_id.tolist()
                    ):
                        raise ValueError("Expanded bank row order differs")
                banks["retrieval"] = bank
                for name, families in plan["variants"].items():

                    def fit(
                        path,
                        families=families,
                        training=training,
                        validation=validation,
                        banks=banks,
                        fold=fold,
                    ):
                        _model(path, training, validation, banks, families, fold)

                    completed = stage(directory, prefix + "_model_" + name, fit)
                    part = pd.read_csv(completed / "predictions.csv")
                    part["model"], part["protocol"] = name, protocol
                    parts.append(part)
                    details = json.loads((completed / "details.json").read_text())
                    importance.extend(
                        {"protocol": protocol, "fold": fold, "model": name, **item}
                        for item in details["importance"]
                    )
                log.emit("retrieval_fold_completed", protocol=protocol, fold=fold)
        oof = pd.concat(parts, ignore_index=True)

        def review(path):
            results, uncertainty = [], []
            for protocol in splits:
                references = {"rule_examples", *plan["matching_controls"].values()}
                scores = {
                    name: aligned_predictions(
                        frame, baseline[(baseline.protocol == protocol) & (baseline.model == name)]
                    )
                    for name in references
                }
                for name, part in oof[oof.protocol == protocol].groupby("model"):
                    scores[name] = aligned_predictions(frame, part)
                    results.append(
                        {
                            "protocol": protocol,
                            "model": name,
                            "metrics": evaluate(frame.rule_violation, scores[name], frame.rule),
                        }
                    )
                contrasts = [(name, name, "rule_examples") for name in plan["variants"]]
                contrasts += [
                    ("add_retrieval_to_" + ref, name, ref)
                    for name, ref in plan["matching_controls"].items()
                ]
                uncertainty.extend(
                    {"protocol": protocol, **item}
                    for item in paired_auc_comparisons(
                        frame.rule_violation,
                        scores,
                        frame.rule,
                        frame.body.map(normalize),
                        contrasts,
                        draws=plan["bootstrap_draws"],
                        seed=plan["seed"],
                    )
                )
            for name, value in {
                "results.json": results,
                "uncertainty.json": uncertainty,
                "screening.json": screens,
                "importance.json": importance,
            }.items():
                atomic_json(path / name, value)
            atomic_bytes(path / "oof.csv", oof.to_csv(index=False).encode())

        stage(directory, "review", review)
        export_retrieval(root, directory)
        log.emit("retrieval_completed", run_id=directory.name, fitted_models=35)
    return directory


def export_retrieval(root: Path, directory: Path) -> None:
    identity = json.loads((directory / "provenance.json").read_text())
    if (
        identity["source_sha256"] != digest(Path(__file__))
        or content_key(identity)[:20] != directory.name
    ):
        raise ValueError("Retrieval source or run identity differs")
    if identity["expanded_metadata_sha256"] != digest(root / "reports/expanded/metadata.json"):
        raise ValueError("Retrieval base evidence differs")
    marker = json.loads((directory / "review/complete.json").read_text())["files"]
    frame = load_development(root)
    for name in [*PUBLIC_FILES, "oof.csv"]:
        if digest(directory / "review" / name) != marker[name]:
            raise ValueError("Retrieval review checksum differs")
    oof = pd.read_csv(directory / "review/oof.csv")
    for record in json.loads((directory / "review/results.json").read_text()):
        values = aligned_predictions(
            frame, oof[(oof.protocol == record["protocol"]) & (oof.model == record["model"])]
        )
        actual = evaluate(frame.rule_violation, values, frame.rule)
        if any(
            abs(actual[m] - record["metrics"][m]) > 1e-12
            for m in ("rule_macro_auc", "log_loss", "brier", "pooled_auc")
        ):
            raise ValueError("Retrieval metrics differ from private predictions")
    public = root / "reports/retrieval"
    for name in PUBLIC_FILES:
        atomic_bytes(public / name, (directory / "review" / name).read_bytes())
    atomic_json(
        public / "metadata.json",
        {
            **identity,
            "run_id": directory.name,
            "private_checkpoint_sha256": digest(directory / "review/complete.json"),
            "files": {name: digest(public / name) for name in PUBLIC_FILES},
        },
    )


def retrieval_evidence(root: Path) -> dict | None:
    path = root / "reports/retrieval/metadata.json"
    if not path.exists():
        return None
    metadata = json.loads(path.read_text())
    expanded_evidence(root)
    if (
        metadata.get("source_sha256") != digest(Path(__file__))
        or metadata.get("plan_sha256") != digest(root / "configs/retrieval.json")
        or metadata.get("expanded_metadata_sha256")
        != digest(root / "reports/expanded/metadata.json")
        or metadata.get("protocol_commit") != PROTOCOL_COMMIT
        or set(metadata.get("files", {})) != set(PUBLIC_FILES)
    ):
        raise ValueError("Stale semantic retrieval evidence")
    identity = {
        k: v
        for k, v in metadata.items()
        if k not in {"run_id", "files", "private_checkpoint_sha256"}
    }
    if content_key(identity)[:20] != metadata["run_id"]:
        raise ValueError("Retrieval evidence identity differs")
    result = {"metadata": metadata}
    for name, sha in metadata["files"].items():
        if digest(path.parent / name) != sha:
            raise ValueError("Retrieval public checksum differs")
        result[name.removesuffix(".json")] = json.loads((path.parent / name).read_text())
    return result
