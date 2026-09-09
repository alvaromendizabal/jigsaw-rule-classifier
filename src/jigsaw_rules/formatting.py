"""Frozen semantic input hypotheses, isolated caches and matched development ablations."""

from __future__ import annotations

import inspect
import json
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from filelock import FileLock
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from jigsaw_rules.data import EXAMPLES, normalize
from jigsaw_rules.diagnostics import support_scores
from jigsaw_rules.embeddings import QwenEncoder, content_key, load_spec, prepare_model
from jigsaw_rules.expanded import (
    aligned_predictions,
    expanded_evidence,
    load_development,
    support_stress,
    validate_splits,
)
from jigsaw_rules.expanded_embeddings import encode_batches, prepare_plan, verified_keys
from jigsaw_rules.metrics import evaluate
from jigsaw_rules.pairs import NLIEncoder, _commit_pair_shard, pair_inputs, prepare_pair_model
from jigsaw_rules.representations import TrainingScreen, semantic_candidates
from jigsaw_rules.research import _model, cached_vectors, read_embedding_cache
from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest, environment, stage
from jigsaw_rules.uncertainty import paired_auc_comparisons

PROTOCOL_COMMIT = "915ce70792adff3b85c0eac5bb7e461a646284aa"
PLAN_SHA256 = "12af2ad1ed2c3577d973c2a75302e1d17144f7609f56effaff571413d88dab1c"
ERROR_AUDIT_SHA256 = "231fb1d5f625c7fd8903f5eefdec73a35b070e0a71d9a4178ed0f94a5c1f3699"
PUBLIC_FILES = (
    "results.json",
    "uncertainty.json",
    "screening.json",
    "importance.json",
    "audit.json",
    "error_audit.json",
)


def load_plan(root: Path) -> dict:
    path = root / "configs/formatting.json"
    if digest(path) != PLAN_SHA256:
        raise ValueError("Semantic formatting specification differs from its pre-score protocol")
    return json.loads(path.read_text())


def behavior_hypothesis(rule: str) -> str:
    """Remove a prohibitory wrapper without changing the described behavior's scope."""
    if not isinstance(rule, str) or ":" not in rule:
        raise ValueError("Unsupported policy syntax; do not guess prohibited behavior")
    description = rule.split(":", 1)[1].strip()
    match = re.fullmatch(r"Do not (.+)\.", description)
    if match:
        return f"The author uses this comment to {match[1]}."
    match = re.fullmatch(r"(.+) are not allowed\.", description)
    if match:
        scope = match[1][0].lower() + match[1][1:]
        return f"This comment contains {scope}."
    raise ValueError("Unsupported policy syntax; do not guess prohibited behavior")


def intent_inputs(frame: pd.DataFrame) -> list[tuple[str, str]]:
    if "rule_violation" in frame:
        raise ValueError("Encoder inputs must not contain query targets")
    return [
        (row.body, hypothesis)
        for row in frame.itertuples()
        for hypothesis in (
            f"This comment violates the following community rule: {row.rule}",
            behavior_hypothesis(row.rule),
        )
    ]


def document_inputs(frame: pd.DataFrame) -> list[str]:
    if "rule_violation" in frame:
        raise ValueError("Encoder inputs must not contain query targets")
    return frame[EXAMPLES].to_numpy().ravel().tolist()


def document_encoder(root: Path) -> QwenEncoder:
    encoder = QwenEncoder(root, load_spec(root))
    encoder.contract = {
        **encoder.contract,
        "input_role": "plain_support_document_v1",
        "formatter_sha256": content_key(inspect.getsource(document_inputs)),
    }
    return encoder


def _document_worker(root, plan, batches, worker):
    return encode_batches(root, plan, batches, document_encoder(root), worker)


def plain_support_vectors(root: Path, frame: pd.DataFrame, *, encode: bool):
    plan = load_plan(root)
    texts = document_inputs(frame)
    encoder = document_encoder(root)
    contract = encoder.contract
    cache = root / "runs/embeddings" / content_key(contract)[:20]
    with FileLock(str(root / "runs/formatting_documents.lock"), timeout=1):
        if encode:
            task_plan = prepare_plan(root, texts, contract, plan["shard_size"])
            batches = json.loads((task_plan / "batches.json").read_text())
            present = verified_keys(cache, contract)
            pending = [b for b in batches if any(content_key(t) not in present for t in b)]
            if pending:
                prepare_model(root, load_spec(root))
                with ProcessPoolExecutor(
                    max_workers=plan["workers"], mp_context=multiprocessing.get_context("spawn")
                ) as pool:
                    futures = [
                        pool.submit(_document_worker, root, task_plan, batch, i)
                        for i in range(plan["workers"])
                        if (batch := pending[i :: plan["workers"]])
                    ]
                    for future in as_completed(futures):
                        future.result()
        array, audit = read_embedding_cache(cache, contract, [content_key(t) for t in texts])
        statistics = [json.loads(p.read_text()) for p in cache.glob("batch_*/statistics.json")]
        audit.update(
            {
                "summed_encoder_seconds": sum(s["encode_seconds"] for s in statistics),
                "truncated_documents": sum(s["truncated_rows"] for s in statistics),
                "maximum_original_tokens": max(s["maximum_original_tokens"] for s in statistics),
                "peak_encoder_process_rss_gib": max(s["peak_rss_gib"] for s in statistics),
            }
        )
    return array.reshape(len(frame), 4, -1), audit


def verified_stage(directory: Path) -> dict:
    if directory.is_symlink():
        raise ValueError("Symlinked stages are not accepted")
    marker = directory / "complete.json"
    files = json.loads(marker.read_text())["files"]
    if not files:
        raise ValueError("An empty stage is not a checkpoint")
    for name, sha in files.items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()) or digest(path) != sha:
            raise ValueError("Stage checksum or path differs")
    return files


def probability_shards(cache: Path) -> tuple[dict, dict]:
    found, markers = {}, {}
    for path in sorted(cache.glob("batch_*")):
        if not path.is_dir():
            continue
        verified_stage(path)
        keys = json.loads((path / "inputs.json").read_text())
        values = np.load(path / "probabilities.npy", allow_pickle=False)
        if (
            values.shape != (len(keys), 3)
            or len(set(keys)) != len(keys)
            or not np.isfinite(values).all()
            or (values < 0).any()
            or (values > 1).any()
            or not np.allclose(values.sum(1), 1, atol=1e-6)
        ):
            raise ValueError("Invalid NLI probability checkpoint")
        for key, value in zip(keys, values, strict=True):
            if key in found and not np.array_equal(value, found[key]):
                raise ValueError("Conflicting NLI inference for one input")
            found[key] = value
        markers[path.name] = digest(path / "complete.json")
    return found, markers


def intent_probabilities(root: Path, frame: pd.DataFrame, *, encode: bool):
    inputs = intent_inputs(frame)
    config = json.loads((root / "configs/pairs.json").read_text())
    software = {p: version(p) for p in ("torch", "transformers", "numpy")}
    legacy = {
        "config": config,
        "batch_order": "combined_character_length_then_input_hash",
        "source_sha256": content_key(
            [inspect.getsource(NLIEncoder), inspect.getsource(pair_inputs)]
        ),
        "software": software,
    }
    contract = {
        **legacy,
        "source_sha256": content_key(
            [
                inspect.getsource(NLIEncoder),
                inspect.getsource(intent_inputs),
                inspect.getsource(behavior_hypothesis),
            ]
        ),
        "input_role": "generic_and_affirmative_policy_v1",
    }
    cache = root / "runs/intent" / content_key(contract)[:20]
    old = root / "runs/pairs" / content_key(legacy)[:20]
    prior, prior_markers = {}, {}
    if (old / "contract.json").exists():
        if json.loads((old / "contract.json").read_text()) != legacy:
            raise ValueError("Historical NLI contract differs")
        prior, prior_markers = probability_shards(old)
    with FileLock(str(root / "runs/formatting_intent.lock"), timeout=1):
        if (cache / "contract.json").exists():
            if json.loads((cache / "contract.json").read_text()) != contract:
                raise ValueError("Policy-intent cache contract differs")
        elif encode:
            atomic_json(cache / "contract.json", contract)
        else:
            raise ValueError("No policy-intent cache; encoding was not requested")
        found, _ = probability_shards(cache)
        unique = {content_key(pair): pair for pair in inputs}
        reusable = sorted(set(unique) & set(prior) - set(found))
        if reusable and encode:

            def reuse(path):
                np.save(path / "probabilities.npy", np.stack([prior[k] for k in reusable]))
                atomic_json(path / "inputs.json", reusable)
                atomic_json(
                    path / "statistics.json",
                    {
                        "seconds": 0,
                        "truncated": 0,
                        "reused_inputs": len(reusable),
                        "legacy_cache": old.name,
                        "legacy_shards_sha256": content_key(prior_markers),
                    },
                )

            stage(cache, "batch_reused_" + content_key(reusable)[:20], reuse)
            found.update({k: prior[k] for k in reusable})
        missing = sorted(set(unique) - set(found), key=lambda k: (sum(map(len, unique[k])), k))
        if missing and not encode:
            raise ValueError("Policy-intent cache is incomplete; no encoding was started")
        if missing:
            prepare_pair_model(root, config)
            batches = [missing[i : i + 64] for i in range(0, len(missing), 64)]
            with ProcessPoolExecutor(
                max_workers=config["inference_workers"],
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                futures = [
                    pool.submit(
                        _commit_pair_shard,
                        str(root),
                        config,
                        str(cache),
                        batch,
                        [unique[k] for k in batch],
                    )
                    for batch in batches
                ]
                for future in as_completed(futures):
                    future.result()
        found, markers = probability_shards(cache)
        statistics = [json.loads(p.read_text()) for p in cache.glob("batch_*/statistics.json")]
        audit = {
            "cache": cache.name,
            "contract_sha256": content_key(contract),
            "shards_sha256": content_key(markers),
            "shards_verified": len(markers),
            "requested_pairs": len(inputs),
            "unique_pairs": len(unique),
            "historical_inputs_reused": sum(s.get("reused_inputs", 0) for s in statistics),
            "new_unique_inputs": sum(
                len(json.loads(p.read_text()))
                for p in cache.glob("batch_*/inputs.json")
                if "batch_reused_" not in p.parent.name
            ),
            "summed_encoder_seconds": sum(s["seconds"] for s in statistics),
            "truncated_new_pairs": sum(s["truncated"] for s in statistics),
        }
        return np.stack([found[content_key(p)] for p in inputs]).reshape(len(frame), 2, 3), audit


def intent_features(probabilities: np.ndarray) -> tuple[np.ndarray, list[str]]:
    p = np.asarray(probabilities, dtype=np.float64)
    if (
        p.ndim != 3
        or p.shape[1:] != (2, 3)
        or not np.isfinite(p).all()
        or (p < 0).any()
        or (p > 1).any()
        or not np.allclose(p.sum(2), 1, atol=1e-6)
    ):
        raise ValueError("Expected two normalized NLI probability triples")
    columns, names = [], []
    for i, name in enumerate(("generic", "behavior")):
        c, e, n = p[:, i].T
        columns.extend(
            [
                c,
                e,
                n,
                e - c,
                np.log((e + 1e-6) / (c + 1e-6)),
                -(p[:, i] * np.log(p[:, i].clip(1e-12))).sum(1),
            ]
        )
        names.extend(
            f"{name}/{s}"
            for s in ("contradiction", "entailment", "neutral", "margin", "log_odds", "entropy")
        )
    columns.extend([p[:, 1, 1] - p[:, 0, 1], p[:, 1, 0] - p[:, 0, 0], p[:, 1, 1] * p[:, 0, 1]])
    names.extend(["entailment_change", "contradiction_change", "entailment_product"])
    base = np.column_stack(columns)
    return np.column_stack([base, np.sign(base) * np.sqrt(np.abs(base)), base * np.abs(base)]), (
        names + [n + "/signed_sqrt" for n in names] + [n + "/signed_square" for n in names]
    )


def asymmetric_vectors(original: np.ndarray, documents: np.ndarray) -> np.ndarray:
    if original.ndim != 3 or original.shape[1] != 5 or documents.shape != original[:, 1:].shape:
        raise ValueError("Asymmetric support geometry is not aligned")
    result = np.concatenate([original[:, :1], documents], axis=1)
    if not np.isfinite(result).all() or not np.allclose(
        np.linalg.norm(result, axis=2), 1, atol=1e-4
    ):
        raise ValueError("Expected finite unit vectors")
    return result


def build_bank(path, training, validation, raw_train, raw_valid, names, budget):
    screen = TrainingScreen(budget).fit(raw_train, training.rule_violation.to_numpy(), names)
    x_train, x_valid = screen.transform(raw_train, names), screen.transform(raw_valid, names)
    if not x_train.shape[1]:
        raise ValueError("All semantic features were screened out")
    scaler = StandardScaler().fit(x_train)
    for name, array in (("train", x_train), ("valid", x_valid)):
        sparse.save_npz(
            path / f"{name}.npz",
            sparse.csr_matrix(scaler.transform(array) / np.sqrt(array.shape[1])),
        )
    joblib.dump((screen, scaler), path / "screen.joblib", compress=3)
    atomic_bytes(path / "catalog.csv", screen.catalog_.to_csv(index=False).encode())
    atomic_json(path / "selected.json", [names[i] for i in screen.indices_])
    atomic_json(
        path / "details.json",
        {
            **screen.audit_,
            "train_row_ids_sha256": content_key(training.row_id.tolist()),
            "validation_row_ids_sha256": content_key(validation.row_id.tolist()),
        },
    )


def original_predictions(root: Path, frame: pd.DataFrame, base: Path) -> pd.DataFrame:
    verified_stage(base / "review")
    oof = pd.read_csv(base / "review/oof.csv")
    for (_protocol, model), part in oof.groupby(["protocol", "model"]):
        if model in {"qwen_centroid", "semantic_scalar_only", "word_semantic_scalar"}:
            aligned_predictions(frame, part)
    return oof


def sample_errors(frame: pd.DataFrame, probabilities: np.ndarray, plan: dict) -> pd.DataFrame:
    records = []
    data = frame.copy()
    data["reference_probability"] = probabilities
    data["body_key"] = data.body.map(normalize)
    for rule in plan["error_audit"]["policies"]:
        for label in (0, 1):
            group = data[(data.rule == rule) & (data.rule_violation == label)].copy()
            group = group.sort_values(["reference_probability", "row_id"]).drop_duplicates(
                "body_key"
            )
            group["score_tertile"] = np.minimum(np.arange(len(group)) * 3 // len(group), 2)
            group["sample_key"] = group.body_key.map(lambda b: content_key([plan["seed"], b]))
            for tertile in (0, 1, 2):
                part = group[group.score_tertile == tertile].sort_values("sample_key").head(4)
                if len(part) != 4:
                    raise ValueError("Insufficient distinct bodies for the frozen error audit")
                records.append(part)
    return pd.concat(records, ignore_index=True)


def run_formatting(root: Path, *, encode: bool = False) -> Path:
    plan, base_evidence = load_plan(root), expanded_evidence(root)
    error_path = root / "reports/formatting/error_audit.json"
    if digest(error_path) != ERROR_AUDIT_SHA256:
        raise ValueError("The frozen pre-score error audit is required")
    error_audit = json.loads(error_path.read_text())
    if base_evidence is None or base_evidence["metadata"]["run_id"] != plan["expanded_run"]:
        raise ValueError("The matched expanded study is required")
    frame = load_development(root)
    base = root / "runs/expanded" / plan["expanded_run"]
    verified_stage(base / "design")
    if json.loads((base / "design/row_ids.json").read_text()) != frame.row_id.tolist():
        raise ValueError("Development rows differ from saved splits")
    splits = json.loads((base / "design/splits.json").read_text())
    validate_splits(frame, splits)
    original = original_predictions(root, frame, base)
    vectors, original_cache = cached_vectors(root, frame)
    query = frame.drop(columns="rule_violation")
    with Progress(root / "logs/formatting.jsonl", "semantic_formatting_study") as log:
        documents, document_cache = plain_support_vectors(root, query, encode=encode)
        probabilities, nli_cache = intent_probabilities(root, query, encode=encode)
        identity = {
            "schema": 1,
            "protocol_commit": PROTOCOL_COMMIT,
            "plan_sha256": digest(root / "configs/formatting.json"),
            "source_sha256": digest(Path(__file__)),
            "error_audit_sha256": ERROR_AUDIT_SHA256,
            "nli_source_sha256": digest(root / "src/jigsaw_rules/pairs.py"),
            "expanded_metadata_sha256": digest(root / "reports/expanded/metadata.json"),
            "original_cache": original_cache,
            "document_cache": document_cache,
            "nli_cache": nli_cache,
            "environment": environment(),
            "confirmation_targets_accessed": False,
        }
        directory = root / "runs/formatting" / content_key(identity)[:20]
        atomic_json(directory / "provenance.json", identity)
        asymmetric = asymmetric_vectors(vectors, documents)
        raw, names = semantic_candidates(asymmetric)
        indices = [i for i, n in enumerate(names) if n.startswith("similarity/")]
        candidates = {
            "asymmetric_scalar": (raw[:, indices], [names[i] for i in indices]),
            "intent_scalar": intent_features(probabilities),
        }
        frozen = {
            "asymmetric_centroid": support_scores(asymmetric)["qwen_centroid"],
            "generic_entailment": probabilities[:, 0, 1],
            "behavior_entailment": probabilities[:, 1, 1],
        }
        parts, screens, importance, selections = [], [], [], {}
        for protocol, records in splits.items():
            for fold, assignment in enumerate(records):
                ti, vi = assignment["train"], assignment["valid"]
                training, validation = frame.iloc[ti], frame.iloc[vi]
                prefix, banks = f"{protocol}_{fold}", {}
                for family in ("word", "semantic_scalar"):
                    path = base / f"{prefix}_{family}"
                    verified_stage(path)
                    details = json.loads((path / "details.json").read_text())
                    if details["train_row_ids_sha256"] != content_key(
                        training.row_id.tolist()
                    ) or details["validation_row_ids_sha256"] != content_key(
                        validation.row_id.tolist()
                    ):
                        raise ValueError("Restored feature bank row alignment differs")
                    banks[family] = path
                for family, (array, names) in candidates.items():
                    path = stage(
                        directory,
                        f"{prefix}_{family}",
                        partial(
                            build_bank,
                            training=training,
                            validation=validation,
                            raw_train=array[ti],
                            raw_valid=array[vi],
                            names=names,
                            budget=plan["budgets"][family],
                        ),
                    )
                    banks[family] = path
                    screens.append(
                        {
                            "protocol": protocol,
                            "fold": fold,
                            "family": family,
                            **json.loads((path / "details.json").read_text()),
                        }
                    )
                    selections[(protocol, fold, family)] = set(
                        json.loads((path / "selected.json").read_text())
                    )
                for model, families in plan["fitted_variants"].items():
                    path = stage(
                        directory,
                        f"{prefix}_{model}_model",
                        partial(
                            _model,
                            training=training,
                            validation=validation,
                            banks=banks,
                            families=tuple(families),
                            fold=fold,
                        ),
                    )
                    part = pd.read_csv(path / "predictions.csv").assign(
                        protocol=protocol, model=model
                    )
                    if part.row_id.tolist() != validation.row_id.tolist():
                        raise ValueError("Saved fitted predictions differ from the validation rows")
                    parts.append(part)
                    importance.extend(
                        {"protocol": protocol, "fold": fold, "model": model, **item}
                        for item in json.loads((path / "details.json").read_text())["importance"]
                    )
                for model, score in frozen.items():
                    part = validation[["row_id", "rule", "rule_violation"]].copy()
                    parts.append(
                        part.assign(
                            probability=score[vi], fold=fold, protocol=protocol, model=model
                        )
                    )
                log.emit("fold_completed", protocol=protocol, fold=fold)
        oof = pd.concat(
            [
                *parts,
                original[
                    original.model.isin(
                        ["qwen_centroid", "semantic_scalar_only", "word_semantic_scalar"]
                    )
                ],
            ],
            ignore_index=True,
        )

        def review(path):
            results = [
                {
                    "protocol": protocol,
                    "model": model,
                    "metrics": evaluate(
                        frame.rule_violation, aligned_predictions(frame, part), frame.rule
                    ),
                }
                for (protocol, model), part in oof.groupby(["protocol", "model"])
            ]
            transfer = {
                model: aligned_predictions(frame, part)
                for model, part in oof[oof.protocol == "heldout_rule"].groupby("model")
            }
            uncertainty = paired_auc_comparisons(
                frame.rule_violation,
                transfer,
                frame.rule,
                frame.body.map(normalize),
                [tuple(c) for c in plan["contrasts"]],
                draws=plan["bootstrap_draws"],
                seed=plan["seed"],
            )
            stability = [
                {
                    "family": family,
                    "fold_a": [a[0], a[1]],
                    "fold_b": [b[0], b[1]],
                    "jaccard": len(selections[a] & selections[b])
                    / len(selections[a] | selections[b]),
                }
                for family in candidates
                for a, b in combinations([key for key in selections if key[2] == family], 2)
            ]
            audit = {
                "development_rows": len(frame),
                "development_policies": frame.rule.nunique(),
                "fitted_models": sum(map(len, splits.values())) * len(plan["fitted_variants"]),
                "frozen_scores": 3,
                "candidate_counts": {
                    family: len(names) for family, (_, names) in candidates.items()
                },
                "stability": stability,
                "asymmetric_support_stress": support_stress(frame, asymmetric, plan["seed"]),
                "hypotheses": {r: behavior_hypothesis(r) for r in sorted(frame.rule.unique())},
                "confirmation_targets_accessed": False,
            }
            for name, payload in {
                "results.json": results,
                "uncertainty.json": uncertainty,
                "screening.json": screens,
                "importance.json": importance,
                "audit.json": audit,
                "error_audit.json": error_audit,
            }.items():
                atomic_json(path / name, payload)
            atomic_bytes(path / "oof.csv", oof.to_csv(index=False).encode())

        path = stage(directory, "review", review)
        verified_stage(path)
        public = root / "reports/formatting"
        for name in PUBLIC_FILES:
            atomic_bytes(public / name, (path / name).read_bytes())
        atomic_json(
            public / "metadata.json",
            {
                **identity,
                "run_id": directory.name,
                "private_checkpoint_sha256": digest(path / "complete.json"),
                "files": {name: digest(public / name) for name in PUBLIC_FILES},
            },
        )
        return directory


def formatting_evidence(root: Path) -> dict | None:
    path = root / "reports/formatting/metadata.json"
    if not path.exists():
        return None
    metadata = json.loads(path.read_text())
    expanded_evidence(root)
    if (
        metadata.get("protocol_commit") != PROTOCOL_COMMIT
        or metadata.get("error_audit_sha256") != ERROR_AUDIT_SHA256
        or metadata.get("plan_sha256") != digest(root / "configs/formatting.json")
        or metadata.get("source_sha256") != digest(Path(__file__))
        or metadata.get("nli_source_sha256") != digest(root / "src/jigsaw_rules/pairs.py")
        or metadata.get("expanded_metadata_sha256")
        != digest(root / "reports/expanded/metadata.json")
        or set(metadata.get("files", {})) != set(PUBLIC_FILES)
    ):
        raise ValueError("Stale semantic formatting evidence")
    identity = {
        k: v
        for k, v in metadata.items()
        if k not in {"run_id", "files", "private_checkpoint_sha256"}
    }
    if content_key(identity)[:20] != metadata["run_id"]:
        raise ValueError("Semantic formatting identity differs")
    result = {"metadata": metadata}
    for name, sha in metadata["files"].items():
        if digest(path.parent / name) != sha:
            raise ValueError("Semantic formatting public checksum differs")
        result[name.removesuffix(".json")] = json.loads((path.parent / name).read_text())
    return result
