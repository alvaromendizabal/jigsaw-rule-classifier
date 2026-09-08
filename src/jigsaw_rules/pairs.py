"""Frozen joint-text NLI features with pinned assets and resumable CPU inference."""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from filelock import FileLock
from scipy import sparse
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from jigsaw_rules.data import EXAMPLES, load_data, normalize
from jigsaw_rules.embeddings import content_key
from jigsaw_rules.metrics import evaluate
from jigsaw_rules.representations import TrainingScreen
from jigsaw_rules.research import CORE, research_evidence, verify_research
from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest, stage
from jigsaw_rules.semantic import reference_splits
from jigsaw_rules.uncertainty import paired_auc_comparisons

VARIANTS = ("rule_nli", "support_nli", "all_nli", "word_nli", "transfer_nli")
PUBLIC_FILES = ("results.json", "uncertainty.json", "screening.json", "inference.json")


def pair_inputs(frame: pd.DataFrame) -> list[tuple[str, str]]:
    """The two templates are declared before evaluation; no label-dependent prompts."""
    return [
        (row.body, hypothesis)
        for row in frame.itertuples()
        for hypothesis in (
            f"This comment violates the following community rule: {row.rule}",
            f"This comment follows the following community rule: {row.rule}",
            *(getattr(row, column) for column in EXAMPLES),
        )
    ]


def pair_candidates(probabilities: np.ndarray) -> tuple[np.ndarray, list[str]]:
    if (
        probabilities.ndim != 3
        or probabilities.shape[1:] != (6, 3)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(axis=2), 1)
    ):
        raise ValueError("Expected six aligned three-class NLI probability vectors")
    arrays, names = [], []
    for label, group in enumerate(("contradiction", "entailment", "neutral")):
        rule = probabilities[:, :2, label]
        arrays.extend([rule[:, 0], rule[:, 1], rule[:, 0] - rule[:, 1]])
        names.extend(f"rule/{group}/{name}" for name in ("violation", "compliance", "margin"))
        positive = np.sort(probabilities[:, 2:4, label], axis=1)
        negative = np.sort(probabilities[:, 4:6, label], axis=1)
        arrays.extend(
            [
                positive[:, 0],
                positive[:, 1],
                negative[:, 0],
                negative[:, 1],
                positive[:, 1] - negative[:, 1],
                positive.mean(1) - negative.mean(1),
                positive[:, 1] - positive[:, 0],
                negative[:, 1] - negative[:, 0],
            ]
        )
        names.extend(
            f"support/{group}/{name}"
            for name in (
                "positive_min",
                "positive_max",
                "negative_min",
                "negative_max",
                "maximum_margin",
                "mean_margin",
                "positive_spread",
                "negative_spread",
            )
        )
    base = np.column_stack(arrays)
    return np.column_stack([base, np.sign(base) * np.sqrt(np.abs(base)), base * np.abs(base)]), (
        names
        + [f"{name}/signed_sqrt" for name in names]
        + [f"{name}/signed_square" for name in names]
    )


def prepare_pair_model(root: Path, config: dict) -> Path:
    destination = root / "models/nli-deberta-v3-small" / config["revision"]
    if any(not (destination / name).is_file() for name in config["files"]):
        subprocess.run(
            [
                str(Path(sys.executable).with_name("hf")),
                "download",
                config["model_id"],
                "--revision",
                config["revision"],
                "--include",
                *config["files"],
                "--local-dir",
                str(destination),
            ],
            check=True,
            timeout=1200,
        )
    for name, record in config["files"].items():
        payload = (destination / name).read_bytes()
        actual = (
            hashlib.sha256(payload).hexdigest()
            if record["algorithm"] == "sha256"
            else hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        )
        if actual != record["digest"]:
            raise ValueError(f"Pinned NLI asset checksum mismatch: {name}")
    return destination


class NLIEncoder:
    def __init__(self, root: Path, config: dict):
        self.root, self.config = root, config
        self.model = self.tokenizer = None
        self.truncated = 0

    def encode(self, pairs: list[tuple[str, str]], log: Progress) -> np.ndarray:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self.model is None:
            directory = prepare_pair_model(self.root, self.config)
            torch.set_num_threads(self.config["threads"])
            torch.manual_seed(self.config["seed"])
            torch.use_deterministic_algorithms(True)
            self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                directory, local_files_only=True, use_safetensors=True, trust_remote_code=False
            ).eval()
            if self.model.config.id2label != {0: "contradiction", 1: "entailment", 2: "neutral"}:
                raise ValueError("Unexpected NLI label mapping")
            log.emit("pinned_nli_loaded", revision=self.config["revision"])
        outputs = []
        for start in range(0, len(pairs), self.config["batch_size"]):
            batch = pairs[start : start + self.config["batch_size"]]
            left, right = zip(*batch, strict=True)
            full = self.tokenizer(list(left), list(right), truncation=False)
            self.truncated += sum(len(ids) > self.config["max_length"] for ids in full["input_ids"])
            inputs = self.tokenizer(
                list(left),
                list(right),
                padding=True,
                truncation=True,
                max_length=self.config["max_length"],
                return_tensors="pt",
            )
            with torch.inference_mode():
                probability = self.model(**inputs).logits.softmax(dim=1).cpu().numpy()
            outputs.append(probability)
        return np.concatenate(outputs)


@lru_cache(maxsize=1)
def _worker_encoder(root: str, config: str):
    return NLIEncoder(Path(root), json.loads(config))


def _commit_pair_shard(root, config, cache, selected, inputs):
    encoder = _worker_encoder(str(root), json.dumps(config, sort_keys=True))
    cache = Path(cache)

    def build(path):
        before, prior = time.monotonic(), encoder.truncated
        with Progress(path / "events.jsonl", "nli_inference") as log:
            values = encoder.encode(inputs, log)
        np.save(path / "probabilities.npy", values, allow_pickle=False)
        atomic_json(path / "inputs.json", selected)
        atomic_json(
            path / "statistics.json",
            {"seconds": time.monotonic() - before, "truncated": encoder.truncated - prior},
        )

    return str(stage(cache, "batch_" + content_key(selected)[:20], build))


def cached_pair_probabilities(root: Path, frame: pd.DataFrame, config: dict, log: Progress):
    inputs = pair_inputs(frame)
    contract = {
        "config": config,
        "batch_order": "combined_character_length_then_input_hash",
        "source_sha256": content_key(
            [inspect.getsource(NLIEncoder), inspect.getsource(pair_inputs)]
        ),
        "software": {p: version(p) for p in ("torch", "transformers", "numpy")},
    }
    cache = root / "runs/pairs" / content_key(contract)[:20]
    atomic_json(cache / "contract.json", contract)
    unique = {content_key(pair): pair for pair in inputs}
    keys = sorted(unique, key=lambda key: (sum(map(len, unique[key])), key))
    batches = [keys[start : start + 64] for start in range(0, len(keys), 64)]
    tasks = [
        (str(root), config, str(cache), selected, [unique[key] for key in selected])
        for selected in batches
    ]
    found, timings, truncated = {}, [], 0

    def collect(location):
        nonlocal truncated
        path = Path(location)
        selected = json.loads((path / "inputs.json").read_text())
        values = np.load(path / "probabilities.npy", allow_pickle=False)
        if values.shape != (len(selected), 3) or not np.isfinite(values).all():
            raise ValueError("Invalid NLI checkpoint shape or values")
        found.update(zip(selected, values, strict=True))
        detail = json.loads((path / "statistics.json").read_text())
        timings.append(detail["seconds"])
        truncated += detail["truncated"]
        log.emit("nli_batch_committed", completed=len(found), total=len(keys))

    workers = config.get("inference_workers", 1)
    if workers == 1:
        for task in tasks:
            collect(_commit_pair_shard(*task))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            pending = [pool.submit(_commit_pair_shard, *task) for task in tasks]
            for future in as_completed(pending):
                collect(future.result())
    return np.stack([found[content_key(pair)] for pair in inputs]).reshape(len(frame), 6, 3), {
        "cache": cache.name,
        "unique_pairs": len(keys),
        "requested_pairs": len(inputs),
        "original_inference_seconds": sum(timings),
        "truncated_pairs": truncated,
        "contract": contract,
    }


def run_pairs(root: Path) -> Path:
    root = root.resolve()
    config = json.loads((root / "configs/pairs.json").read_text())
    evidence = research_evidence(root)
    if evidence is None:
        raise ValueError("Publish verified broad feature research before joint-text comparisons")
    broad = root / "runs" / evidence["metadata"]["run_id"]
    verify_research(root, broad)
    train, _, _ = load_data(root / "data/raw")
    if (root / "data/raw/SYNTHETIC.txt").exists():
        raise ValueError("Competition pair research rejects synthetic data")
    baseline_run = evidence["metadata"]["baseline_run"]
    splits = reference_splits(root, train, baseline_run)
    identity = {
        "config": config,
        "source_sha256": digest(Path(__file__)),
        "broad_run": broad.name,
        "training_sha256": digest(root / "data/raw/train.csv"),
        "variants": VARIANTS,
        "environment": {p: version(p) for p in ("torch", "transformers", "numpy", "scikit-learn")},
    }
    directory = root / "runs" / content_key(identity)[:20]
    with (
        FileLock(str(root / "runs/pair_research.lock"), timeout=1),
        Progress(root / "logs/pairs.jsonl", "pair_feature_research") as log,
    ):
        atomic_json(directory / "provenance.json", identity)
        probabilities, inference = cached_pair_probabilities(root, train, config, log)
        candidates, names = pair_candidates(probabilities)
        frames, screens = [], []
        for protocol, assignments in splits.items():
            for fold, assignment in enumerate(assignments):
                training, validation = (
                    train.iloc[assignment["train"]],
                    train.iloc[assignment["valid"]],
                )
                for variant in VARIANTS:

                    def fit(
                        path,
                        variant=variant,
                        assignment=assignment,
                        training=training,
                        validation=validation,
                        protocol=protocol,
                        fold=fold,
                    ):
                        indices = [
                            i
                            for i, name in enumerate(names)
                            if variant not in {"rule_nli", "support_nli"}
                            or name.startswith("rule/" if variant == "rule_nli" else "support/")
                        ]
                        raw_train = candidates[assignment["train"]][:, indices]
                        raw_valid = candidates[assignment["valid"]][:, indices]
                        selected_names = [names[i] for i in indices]
                        screen = TrainingScreen(32).fit(
                            raw_train, training.rule_violation, selected_names
                        )
                        x_train = screen.transform(raw_train, selected_names)
                        x_valid = screen.transform(raw_valid, selected_names)
                        scaler = StandardScaler().fit(x_train)
                        x_train = sparse.csr_matrix(
                            scaler.transform(x_train) / np.sqrt(x_train.shape[1])
                        )
                        x_valid = sparse.csr_matrix(
                            scaler.transform(x_valid) / np.sqrt(x_valid.shape[1])
                        )
                        families = (
                            CORE
                            if variant == "transfer_nli"
                            else (("word",) if variant == "word_nli" else ())
                        )
                        if families:
                            x_train = sparse.hstack(
                                [
                                    sparse.load_npz(broad / f"{protocol}_{fold}_{f}/train.npz")
                                    for f in families
                                ]
                                + [x_train],
                                format="csr",
                            )
                            x_valid = sparse.hstack(
                                [
                                    sparse.load_npz(broad / f"{protocol}_{fold}_{f}/valid.npz")
                                    for f in families
                                ]
                                + [x_valid],
                                format="csr",
                            )
                        model = LogisticRegression(
                            C=2, solver="liblinear", max_iter=2000, random_state=2025
                        )
                        model.fit(x_train, training.rule_violation)
                        part = validation[["row_id", "rule", "rule_violation"]].copy()
                        part["probability"] = model.predict_proba(x_valid)[:, 1]
                        part["protocol"], part["fold"], part["model"] = protocol, fold, variant
                        atomic_bytes(path / "predictions.csv", part.to_csv(index=False).encode())
                        atomic_json(
                            path / "screening.json",
                            {
                                **screen.audit_,
                                "selected": [selected_names[i] for i in screen.indices_],
                            },
                        )
                        atomic_bytes(
                            path / "catalog.csv", screen.catalog_.to_csv(index=False).encode()
                        )
                        joblib.dump((model, screen, scaler), path / "model.joblib", compress=3)

                    path = stage(directory, f"{protocol}_{fold}_{variant}", fit)
                    frames.append(pd.read_csv(path / "predictions.csv"))
                    screens.append(
                        {
                            "protocol": protocol,
                            "fold": fold,
                            "model": variant,
                            **json.loads((path / "screening.json").read_text()),
                        }
                    )
        # A predeclared label-free rule margin is a diagnostic, not a calibrated probability.
        zero_shot = expit(5 * (probabilities[:, 0, 1] - probabilities[:, 1, 1]))
        for protocol in splits:
            part = train[["row_id", "rule", "rule_violation"]].copy()
            part["probability"], part["protocol"], part["model"] = (
                zero_shot,
                protocol,
                "rule_margin",
            )
            frames.append(part)
        oof = pd.concat(frames, ignore_index=True)
        results, intervals = [], []
        reference = pd.read_csv(root / f"runs/{baseline_run}/review/oof.csv")
        broad_oof = pd.read_csv(broad / "review/oof.csv")
        for protocol in splits:
            bank = {}
            for variant in (*VARIANTS, "rule_margin"):
                part = oof[(oof.protocol == protocol) & (oof.model == variant)]
                if part.row_id.duplicated().any() or set(part.row_id) != set(train.row_id):
                    raise ValueError("Joint-text OOF coverage differs")
                part = part.set_index("row_id").loc[train.row_id]
                bank[variant] = part.probability.to_numpy()
                results.append(
                    {
                        "protocol": protocol,
                        "model": variant,
                        "metrics": evaluate(train.rule_violation, part.probability, train.rule),
                    }
                )
            for name, source, model in (
                ("original_reference", reference, "rule_examples"),
                ("word_screened", broad_oof, "word_screened"),
                ("all_transfer", broad_oof, "all_transfer"),
            ):
                bank[name] = (
                    source[(source.protocol == protocol) & (source.model == model)]
                    .set_index("row_id")
                    .loc[train.row_id]
                    .probability.to_numpy()
                )
            comparisons = [
                (variant, variant, "original_reference") for variant in (*VARIANTS, "rule_margin")
            ]
            comparisons += [
                ("add_nli_to_word", "word_nli", "word_screened"),
                ("add_nli_to_transfer", "transfer_nli", "all_transfer"),
            ]
            intervals.extend(
                {"protocol": protocol, **item}
                for item in paired_auc_comparisons(
                    train.rule_violation, bank, train.rule, train.body.map(normalize), comparisons
                )
            )

        def review(path):
            atomic_json(path / "results.json", results)
            atomic_json(path / "uncertainty.json", intervals)
            atomic_json(path / "screening.json", screens)
            atomic_json(path / "inference.json", inference)
            atomic_bytes(path / "oof.csv", oof.to_csv(index=False).encode())

        path = stage(directory, "review", review)
        for name in PUBLIC_FILES:
            atomic_bytes(root / "reports/pairs" / name, (path / name).read_bytes())
        atomic_json(
            root / "reports/pairs/metadata.json",
            {
                "schema": 1,
                "data_kind": "competition",
                "run_id": directory.name,
                **identity,
                "files": {name: digest(path / name) for name in PUBLIC_FILES},
            },
        )
        log.emit("PAIR_RESEARCH_COMPLETED", run_id=directory.name, fitted_models=25)
        return directory


def pairs_evidence(root: Path) -> dict | None:
    path = root / "reports/pairs/metadata.json"
    if not path.exists():
        return None
    metadata = json.loads(path.read_text())
    broad = research_evidence(root)
    if (
        metadata.get("data_kind") != "competition"
        or broad is None
        or metadata.get("broad_run") != broad["metadata"]["run_id"]
        or metadata.get("source_sha256") != digest(Path(__file__))
        or metadata.get("config") != json.loads((root / "configs/pairs.json").read_text())
        or set(metadata.get("files", {})) != set(PUBLIC_FILES)
    ):
        raise ValueError("Stale joint-text feature evidence")
    result = {"metadata": metadata}
    for name, sha in metadata["files"].items():
        if digest(path.parent / name) != sha:
            raise ValueError("Joint-text aggregate checksum mismatch")
        result[name.removesuffix(".json")] = json.loads((path.parent / name).read_text())
    return result
