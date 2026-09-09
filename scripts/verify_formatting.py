"""Recompute semantic-format metrics and replay saved transformations without fitting."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from jigsaw_rules import formatting
from jigsaw_rules.diagnostics import support_scores
from jigsaw_rules.metrics import evaluate
from jigsaw_rules.representations import semantic_candidates
from jigsaw_rules.runtime import Progress, digest

if __package__:
    from .verify_research import compare_metrics
else:
    from verify_research import compare_metrics


def verify(root: Path) -> dict:
    evidence, plan = formatting.formatting_evidence(root), formatting.load_plan(root)
    if evidence is None:
        raise ValueError("Complete the semantic formatting study before verification")
    frame = formatting.load_development(root)
    directory = root / "runs/formatting" / evidence["metadata"]["run_id"]
    base = root / "runs/expanded" / plan["expanded_run"]
    files = sum(len(formatting.verified_stage(p.parent)) for p in directory.glob("*/complete.json"))
    if (
        digest(directory / "review/complete.json")
        != evidence["metadata"]["private_checkpoint_sha256"]
    ):
        raise ValueError("Private semantic-format review differs")
    for name, sha in evidence["metadata"]["files"].items():
        if digest(directory / "review" / name) != sha:
            raise ValueError("Private and public semantic-format evidence differ")
    formatting.verified_stage(base / "design")
    splits = json.loads((base / "design/splits.json").read_text())
    formatting.validate_splits(frame, splits)
    oof = pd.read_csv(directory / "review/oof.csv")
    models = {
        *plan["fitted_variants"],
        *plan["fixed_scores"],
        "qwen_centroid",
        "semantic_scalar_only",
        "word_semantic_scalar",
    }
    expected = {(protocol, model) for protocol in splits for model in models}
    actual = [(r["protocol"], r["model"]) for r in evidence["results"]]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Metric coverage differs from the frozen specification")
    for record in evidence["results"]:
        part = oof[(oof.protocol == record["protocol"]) & (oof.model == record["model"])]
        p = formatting.aligned_predictions(frame, part)
        compare_metrics(evaluate(frame.rule_violation, p, frame.rule), record["metrics"])
    original, original_cache = formatting.cached_vectors(root, frame)
    query = frame.drop(columns="rule_violation")
    docs, document_cache = formatting.plain_support_vectors(root, query, encode=False)
    probabilities, nli_cache = formatting.intent_probabilities(root, query, encode=False)
    for name, audit in (
        ("original_cache", original_cache),
        ("document_cache", document_cache),
        ("nli_cache", nli_cache),
    ):
        if audit != evidence["metadata"][name]:
            raise ValueError("Replayed inference cache differs from study provenance")
    vectors = formatting.asymmetric_vectors(original, docs)
    raw, names = semantic_candidates(vectors)
    indices = [i for i, name in enumerate(names) if name.startswith("similarity/")]
    candidates = {
        "asymmetric_scalar": (raw[:, indices], [names[i] for i in indices]),
        "intent_scalar": formatting.intent_features(probabilities),
    }
    fixed = {
        "asymmetric_centroid": support_scores(vectors)["qwen_centroid"],
        "generic_entailment": probabilities[:, 0, 1],
        "behavior_entailment": probabilities[:, 1, 1],
        "qwen_centroid": support_scores(original)["qwen_centroid"],
    }
    model_replays, bank_replays, frozen_replays = 0, 0, 0
    for protocol, folds in splits.items():
        assignment = {
            frame.iloc[i].row_id: fold for fold, rows in enumerate(folds) for i in rows["valid"]
        }
        for model, part in oof[oof.protocol == protocol].groupby("model"):
            if not np.array_equal(part.fold, part.row_id.map(assignment)):
                raise ValueError("Saved prediction fold differs from the validation assignment")
            if model in fixed:
                np.testing.assert_allclose(
                    formatting.aligned_predictions(frame, part), fixed[model], rtol=0, atol=1e-12
                )
                frozen_replays += 1
        for fold, rows in enumerate(folds):
            prefix = f"{protocol}_{fold}"
            for family, (array, names) in candidates.items():
                path = directory / f"{prefix}_{family}"
                screen, scaler = joblib.load(path / "screen.joblib")
                if [names[i] for i in screen.indices_] != json.loads(
                    (path / "selected.json").read_text()
                ):
                    raise ValueError("Saved selected-feature schema differs")
                for partition, indices in (("train", rows["train"]), ("valid", rows["valid"])):
                    selected = screen.transform(array[indices], names)
                    transformed = scaler.transform(selected) / np.sqrt(selected.shape[1])
                    np.testing.assert_allclose(
                        transformed,
                        sparse.load_npz(path / f"{partition}.npz").toarray(),
                        rtol=0,
                        atol=1e-12,
                    )
                    bank_replays += 1
            for model, families in plan["fitted_variants"].items():
                path = directory / f"{prefix}_{model}_model"
                matrices = []
                for family in families:
                    source = directory if family in candidates else base
                    bank = source / f"{prefix}_{family}"
                    formatting.verified_stage(bank)
                    matrices.append(sparse.load_npz(bank / "valid.npz"))
                prediction = joblib.load(path / "model.joblib").predict_proba(
                    sparse.hstack(matrices, format="csr")
                )[:, 1]
                saved = pd.read_csv(path / "predictions.csv")
                if saved.row_id.tolist() != frame.iloc[rows["valid"]].row_id.tolist():
                    raise ValueError("Replayed feature bank row order differs")
                np.testing.assert_allclose(prediction, saved.probability, rtol=0, atol=1e-12)
                model_replays += 1
    by_model = {
        r["model"]: r["metrics"]["rule_macro_auc"]
        for r in evidence["results"]
        if r["protocol"] == "heldout_rule"
    }
    for row in evidence["uncertainty"]:
        if (
            abs(row["observed_delta"] - (by_model[row["candidate"]] - by_model[row["reference"]]))
            > 1e-12
        ):
            raise ValueError("Contrast delta differs from recomputed official metrics")
    return {
        "run_id": directory.name,
        "files_verified": files,
        "metric_records": len(actual),
        "model_prediction_replays": model_replays,
        "feature_matrix_replays": bank_replays,
        "frozen_score_replays": frozen_replays,
        "new_encoder_calls": 0,
        "new_model_fits": 0,
        "confirmation_targets_accessed": False,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    with Progress(
        root / "logs/formatting_verification.jsonl", "private_formatting_verification"
    ) as log:
        log.emit("FORMATTING_EVIDENCE_VERIFIED", **verify(root))


if __name__ == "__main__":
    main()
