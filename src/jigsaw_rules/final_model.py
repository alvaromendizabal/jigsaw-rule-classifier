"""Portable selected feature pipeline with exact legacy parity and policy routing."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize as unit_normalize

from jigsaw_rules.calibration import MonotoneSigmoid
from jigsaw_rules.context import ReferenceRanks
from jigsaw_rules.data import EXAMPLES, normalize, validate_frame
from jigsaw_rules.diagnostics import support_scores
from jigsaw_rules.embeddings import content_key
from jigsaw_rules.representations import (
    TrainingScreen,
    lexical_candidates,
    semantic_candidates,
    structural_candidates,
)
from jigsaw_rules.research import BUDGETS, CORE
from jigsaw_rules.runtime import digest


def verify_stage(path: Path) -> dict:
    record = json.loads((path / "complete.json").read_text())
    if not record.get("files"):
        raise ValueError("Empty completed artifact")
    for name, expected in record["files"].items():
        p = path / name
        if (
            p.is_symlink()
            or not p.resolve().is_relative_to(path.resolve())
            or digest(p) != expected
        ):
            raise ValueError("Artifact checksum or path differs")
    return record


def validate_vectors(frame: pd.DataFrame, vectors: np.ndarray, width: int | None = None):
    v = np.asarray(vectors)
    if v.ndim != 3 or v.shape[:2] != (len(frame), 5) or not np.isfinite(v).all():
        raise ValueError("Embedding rows must align with the comment and four supports")
    if width is not None and v.shape[2] != width:
        raise ValueError("Embedding dimension differs from the fitted model")
    if not np.allclose(np.linalg.norm(v, axis=2), 1.0, atol=1e-4):
        raise ValueError("Expected unit-normalized frozen embeddings")
    return v


def semantic_scalars(vectors: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Reuse the original formula in small chunks without retaining coordinate banks."""
    blocks, names = [], None
    for start in range(0, len(vectors), 128):
        values, all_names = semantic_candidates(vectors[start : start + 128])
        indices = [i for i, name in enumerate(all_names) if name.startswith("similarity/")]
        names = [all_names[i] for i in indices]
        blocks.append(values[:, indices])
    if not blocks:
        raise ValueError("Semantic features require nonempty inputs")
    return np.vstack(blocks), names


class FeaturePipeline:
    """The seven selected families; all learned transforms fit training inputs only."""

    def _raw(self, frame, vectors, *, fit_ranks=False):
        word, character = self.word_, self.character_
        lexical = lexical_names = None
        for family in CORE:
            if family in {"word", "character"}:
                vectorizer = word if family == "word" else character
                values = vectorizer.transform(frame.body)
                names = vectorizer.get_feature_names_out().tolist()
            elif family == "structure":
                values, names = structural_candidates(frame)
            elif family in {"lexical", "ranks"}:
                if lexical is None:
                    lexical, lexical_names = lexical_candidates(frame, word, character)
                if family == "lexical":
                    values, names = lexical, lexical_names
                else:
                    if fit_ranks:
                        self.ranker_ = ReferenceRanks().fit(lexical)
                    values = self.ranker_.transform(lexical)
                    names = [f"training_percentile/{name}" for name in lexical_names]
            elif family == "semantic_scalar":
                values, names = semantic_scalars(vectors)
            else:
                support = [word.transform(frame[column]) for column in EXAMPLES]
                direction = (support[0] + support[1] - support[2] - support[3]) / 2
                values = word.transform(frame.body).multiply(direction).tocsr()
                names = [
                    f"comment_x_support_direction/{term}" for term in word.get_feature_names_out()
                ]
            yield family, values, names

    @staticmethod
    def _scale(values, scaler):
        if values.shape[1]:
            if sparse.issparse(values):
                values = unit_normalize(values, norm="l2")
            else:
                values = scaler.transform(values) / np.sqrt(values.shape[1])
        matrix = sparse.csr_matrix(values)
        if not np.isfinite(matrix.data).all():
            raise ValueError("Nonfinite final feature values")
        return matrix

    def fit(self, frame: pd.DataFrame, vectors: np.ndarray, classifier_spec: dict, log=None):
        validate_frame(frame, train=True)
        validate_vectors(frame, vectors)
        corpus = [text for column in ("body", "rule", *EXAMPLES) for text in frame[column]]
        self.word_ = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=60000)
        self.character_ = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=60000
        )
        self.word_.fit(corpus)
        self.character_.fit(corpus)
        self.width_ = vectors.shape[2]
        self.policies_ = tuple(sorted(set(frame.rule.map(normalize))))
        self.screens_, self.audit_, self.selected_ = {}, {}, {}
        matrices = []
        for family, values, names in self._raw(frame, vectors, fit_ranks=True):
            screen = TrainingScreen(
                BUDGETS[family], correlation=None if sparse.issparse(values) else 0.995
            ).fit(values, frame.rule_violation.to_numpy(), names)
            selected = screen.transform(values, names)
            scaler = (
                StandardScaler().fit(selected)
                if not sparse.issparse(selected) and selected.shape[1]
                else None
            )
            matrices.append(self._scale(selected, scaler))
            self.screens_[family] = (screen, scaler)
            self.audit_[family] = screen.audit_
            self.selected_[family] = [names[i] for i in screen.indices_]
            if log:
                log.emit("family_fitted", family=family, **screen.audit_)
        matrix = sparse.hstack(matrices, format="csr")
        if not matrix.shape[1]:
            raise ValueError("No estimable features survived screening")
        self.classifier_ = LogisticRegression(**classifier_spec)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            self.classifier_.fit(matrix, frame.rule_violation)
        return self

    def transform(self, frame: pd.DataFrame, vectors: np.ndarray):
        validate_frame(frame, train=False)
        validate_vectors(frame, vectors, self.width_)
        blocks = []
        for family, values, names in self._raw(frame, vectors):
            screen, scaler = self.screens_[family]
            blocks.append(self._scale(screen.transform(values, names), scaler))
        matrix = sparse.hstack(blocks, format="csr")
        if matrix.shape[1] != self.classifier_.n_features_in_:
            raise ValueError("Final feature order/width differs from the classifier")
        return matrix

    def predict(self, frame: pd.DataFrame, vectors: np.ndarray) -> np.ndarray:
        return self.classifier_.predict_proba(self.transform(frame, vectors))[:, 1]

    @classmethod
    def from_legacy(cls, directory: Path, prefix: str, training: pd.DataFrame, width: int):
        """Load only after checking every required saved stage; never refit a parent."""
        vocabulary = directory / f"{prefix}_vocabulary"
        model = directory / f"{prefix}_model_all_transfer"
        verify_stage(vocabulary)
        verify_stage(model)
        corpus = json.loads((vocabulary / "corpus.json").read_text())
        if corpus["training_row_ids_sha256"] != content_key(training.row_id.tolist()):
            raise ValueError("Loaded parent training identities differ from the routing partition")
        pipeline = cls()
        pipeline.word_, pipeline.character_ = joblib.load(vocabulary / "vocabulary.joblib")
        pipeline.classifier_ = joblib.load(model / "model.joblib")
        pipeline.width_ = width
        pipeline.policies_ = tuple(sorted(set(training.rule.map(normalize))))
        pipeline.screens_, pipeline.audit_, pipeline.selected_ = {}, {}, {}
        for family in CORE:
            path = directory / f"{prefix}_{family}"
            verify_stage(path)
            pipeline.screens_[family] = joblib.load(path / "screen.joblib")
            pipeline.audit_[family] = json.loads((path / "details.json").read_text())
            pipeline.selected_[family] = json.loads((path / "selected.json").read_text())
            if family == "ranks":
                pipeline.ranker_ = joblib.load(path / "ranker.joblib")
        return pipeline


def familiar_mask(frame: pd.DataFrame, training_policies: tuple[str, ...]) -> np.ndarray:
    validate_frame(frame, train=False)
    if not training_policies or len(set(training_policies)) != len(training_policies):
        raise ValueError("A unique nonempty fitted policy set is required")
    if any(normalize(rule) != rule for rule in training_policies):
        raise ValueError("The fitted policy set must be normalized")
    return frame.rule.map(normalize).isin(training_policies).to_numpy()


@dataclass
class RoutedClassifier:
    familiar: FeaturePipeline
    familiar_calibrator: MonotoneSigmoid = field(default_factory=MonotoneSigmoid)
    unseen_calibrator: MonotoneSigmoid = field(default_factory=MonotoneSigmoid)

    def predict(self, frame: pd.DataFrame, vectors: np.ndarray) -> np.ndarray:
        mask = familiar_mask(frame, self.familiar.policies_)
        validate_vectors(frame, vectors, self.familiar.width_)
        output = np.empty(len(frame), dtype=float)
        if mask.any():
            raw = self.familiar.predict(frame.iloc[np.flatnonzero(mask)], vectors[mask])
            output[mask] = self.familiar_calibrator.predict(raw)
        if (~mask).any():
            raw = support_scores(vectors[~mask])["qwen_centroid"]
            output[~mask] = self.unseen_calibrator.predict(raw)
        if not np.isfinite(output).all() or ((output < 0) | (output > 1)).any():
            raise ValueError("Invalid routed predictions")
        return output
