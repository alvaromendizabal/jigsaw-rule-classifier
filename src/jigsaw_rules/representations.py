"""Domain-informed candidates and deterministic, training-only feature screening.

Dense features are hypotheses, not a claim of semantic understanding. Fitted
vocabularies, screening statistics and scaling must never see validation rows.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import sparse

TEXT_STAT_NAMES = (
    "log_characters",
    "log_words",
    "mean_word_length",
    "word_length_std",
    "unique_word_fraction",
    "uppercase_fraction",
    "digit_fraction",
    "punctuation_fraction",
    "whitespace_fraction",
    "log_urls",
    "question_fraction",
    "exclamation_fraction",
    "quotation_fraction",
    "newline_fraction",
    "negation_fraction",
    "first_person_fraction",
    "second_person_fraction",
    "modal_fraction",
    "request_fraction",
    "ellipsis_fraction",
    "empty_text",
)


def text_statistics(text: pd.Series) -> np.ndarray:
    """Only input-text observables; no labels, vocabulary fitting or external lexicons."""
    if not all(isinstance(value, str) for value in text):
        raise ValueError("Text statistics require strings, including explicit empty strings")
    tokens = text.str.findall(r"\b\w+\b")
    characters = text.str.len().to_numpy(dtype=np.float64)
    words = tokens.str.len().to_numpy(dtype=np.float64)
    cden, wden = np.maximum(characters, 1), np.maximum(words, 1)
    lengths = [[len(token) for token in row] for row in tokens]
    columns = [
        np.log1p(characters),
        np.log1p(words),
        np.asarray([np.mean(row) if row else 0.0 for row in lengths]),
        np.asarray([np.std(row) if row else 0.0 for row in lengths]),
        np.asarray([len({word.casefold() for word in row}) for row in tokens]) / wden,
        text.str.count(r"[A-Z]").to_numpy() / cden,
        text.str.count(r"\d").to_numpy() / cden,
        text.str.count(r"[^\w\s]").to_numpy() / cden,
        text.str.count(r"\s").to_numpy() / cden,
        np.log1p(text.str.count(r"(?i)https?://|www\.").to_numpy()),
        text.str.count(r"\?").to_numpy() / cden,
        text.str.count(r"!").to_numpy() / cden,
        text.str.count(r'["“”]|^>').to_numpy() / cden,
        text.str.count(r"\n").to_numpy() / cden,
    ]
    for pattern in (
        r"(?i)\b(?:no|not|never|without)\b|n't",
        r"(?i)\b(?:i|me|my|mine|we|our|us)\b",
        r"(?i)\b(?:you|your|yours)\b",
        r"(?i)\b(?:can|could|should|would|must|might|may)\b",
        r"(?i)\b(?:please|help|recommend|advice|suggest)\b",
    ):
        columns.append(text.str.count(pattern).to_numpy() / wden)
    columns.extend([text.str.count(r"\.{3,}|…").to_numpy() / cden, characters == 0])
    result = np.column_stack(columns).astype(np.float64)
    if result.shape[1] != len(TEXT_STAT_NAMES) or not np.isfinite(result).all():
        raise ValueError("Invalid structural feature schema")
    return result


def structural_candidates(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Order-invariant support summaries, relative structure and motivated interactions."""
    from jigsaw_rules.data import EXAMPLES

    body, rule = text_statistics(frame.body), text_statistics(frame.rule)
    support = np.stack([text_statistics(frame[name]) for name in EXAMPLES], axis=1)
    positive, negative = support[:, :2].mean(axis=1), support[:, 2:].mean(axis=1)
    blocks = {
        "body": body,
        "rule": rule,
        "positive_mean": positive,
        "negative_mean": negative,
        "positive_spread": np.abs(support[:, 0] - support[:, 1]),
        "negative_spread": np.abs(support[:, 2] - support[:, 3]),
        "body_minus_positive": body - positive,
        "body_minus_negative": body - negative,
        "positive_minus_negative": positive - negative,
        "body_minus_rule": body - rule,
    }
    values = list(blocks.values())
    names = [f"{group}/{name}" for group in blocks for name in TEXT_STAT_NAMES]
    original = np.column_stack(values)
    original_names = names.copy()
    values.extend([np.sign(original) * np.sqrt(np.abs(original)), original * np.abs(original)])
    names.extend(f"signed_sqrt/{name}" for name in original_names)
    names.extend(f"signed_square/{name}" for name in original_names)
    for group, comparison in (("support_direction", positive - negative), ("rule", rule)):
        values.append((body[:, :, None] * comparison[:, None, :]).reshape(len(frame), -1))
        names.extend(
            f"body_x_{group}/{left}/{right}"
            for left in TEXT_STAT_NAMES
            for right in TEXT_STAT_NAMES
        )
    return np.column_stack(values), names


def lexical_candidates(frame: pd.DataFrame, word, character) -> tuple[np.ndarray, list[str]]:
    """Provided-example similarities; the caller has fitted both vocabularies on training."""
    from jigsaw_rules.data import EXAMPLES

    values, names = [], []
    statistics = (
        "positive_min",
        "positive_max",
        "negative_min",
        "negative_max",
        "maximum_margin",
        "mean_margin",
        "positive_spread",
        "negative_spread",
    )
    for group, vectorizer in (("word", word), ("character", character)):
        body = vectorizer.transform(frame.body)
        scores = np.column_stack(
            [
                np.asarray(body.multiply(vectorizer.transform(frame[name])).sum(axis=1)).ravel()
                for name in EXAMPLES
            ]
        )
        positive, negative = np.sort(scores[:, :2], axis=1), np.sort(scores[:, 2:], axis=1)
        values.append(
            np.column_stack(
                [
                    positive,
                    negative,
                    positive[:, 1] - negative[:, 1],
                    positive.mean(axis=1) - negative.mean(axis=1),
                    positive[:, 1] - positive[:, 0],
                    negative[:, 1] - negative[:, 0],
                ]
            )
        )
        values.append(np.asarray(body.multiply(vectorizer.transform(frame.rule)).sum(axis=1)))
        names.extend(f"{group}/{name}" for name in (*statistics, "rule_similarity"))
    original = np.column_stack(values)
    original_names = names.copy()
    pairs = list(combinations(range(len(names)), 2))
    values = [original, np.sign(original) * np.sqrt(np.abs(original)), original * np.abs(original)]
    names.extend(f"signed_sqrt/{name}" for name in original_names)
    names.extend(f"signed_square/{name}" for name in original_names)
    values.append(np.column_stack([original[:, i] * original[:, j] for i, j in pairs]))
    names.extend(f"product/{original_names[i]}/{original_names[j]}" for i, j in pairs)
    return np.column_stack(values), names


def semantic_candidates(vectors: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Frozen-embedding geometry, invariant to positive/negative example ordering."""
    from jigsaw_rules.semantic import FEATURE_NAMES, features

    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 3 or vectors.shape[1] != 5 or not np.isfinite(vectors).all():
        raise ValueError("Expected finite aligned body and four support vectors")
    if not np.allclose(np.linalg.norm(vectors, axis=2), 1.0, atol=1e-4):
        raise ValueError("Semantic inputs must have unit L2 norm")
    body = vectors[:, 0]
    positive, negative = vectors[:, 1:3].mean(axis=1), vectors[:, 3:5].mean(axis=1)
    direction = positive - negative
    groups = {
        "comment": body,
        "support_direction": direction,
        "comment_x_direction": body * direction,
        "relative_absolute_distance": np.abs(body - positive) - np.abs(body - negative),
        "positive_spread": np.abs(vectors[:, 1] - vectors[:, 2]),
        "negative_spread": np.abs(vectors[:, 3] - vectors[:, 4]),
    }
    names = [f"{group}/dimension_{i:04d}" for group in groups for i in range(body.shape[1])]
    scalar = features(vectors)
    scalar_names = list(FEATURE_NAMES)
    pnorm = np.linalg.norm(positive, axis=1).clip(1e-12)
    nnorm = np.linalg.norm(negative, axis=1).clip(1e-12)
    pcos = np.einsum("nd,nd->n", body, positive) / pnorm
    ncos = np.einsum("nd,nd->n", body, negative) / nnorm
    extra = [
        pcos,
        ncos,
        pcos - ncos,
        pnorm,
        nnorm,
        pnorm - nnorm,
        np.linalg.norm(direction, axis=1),
        np.einsum("nd,nd->n", positive, negative) / (pnorm * nnorm),
    ]
    scalar_names.extend(
        [
            "positive_centroid_cosine",
            "negative_centroid_cosine",
            "centroid_margin",
            "positive_coherence",
            "negative_coherence",
            "coherence_margin",
            "support_separation",
            "support_centroid_cosine",
        ]
    )
    # Kernel contrasts are fixed feature maps, not validation-tuned temperatures.
    similarities = np.einsum("nd,nkd->nk", body, vectors[:, 1:])
    for temperature in (0.05, 0.1, 0.2, 0.5):
        kernel = np.exp((similarities - 1) / temperature)
        positive_kernel, negative_kernel = kernel[:, :2].mean(1), kernel[:, 2:].mean(1)
        extra.extend(
            [
                positive_kernel,
                negative_kernel,
                positive_kernel - negative_kernel,
                positive_kernel / (positive_kernel + negative_kernel + 1e-12),
            ]
        )
        scalar_names.extend(
            f"kernel_{temperature}/{name}"
            for name in ("positive", "negative", "margin", "positive_share")
        )
    names.extend(f"similarity/{name}" for name in scalar_names)
    return np.column_stack([*groups.values(), scalar, *extra]), names


class TrainingScreen:
    """Deterministic training-only ranking, with explicit rejection reasons.

    Dense correlation checks inspect at most four times the retained budget.
    Sparse families receive exact-duplicate checks, not quadratic correlation scans.
    """

    def __init__(self, maximum: int, correlation: float | None = 0.995):
        if type(maximum) is not int or maximum < 1:
            raise ValueError("A positive integer feature budget is required")
        if correlation is not None and not 0 < correlation <= 1:
            raise ValueError("Correlation threshold must lie in (0, 1]")
        self.maximum, self.correlation = maximum, correlation

    @staticmethod
    def _validate(matrix, names):
        if sparse.issparse(matrix):
            matrix = sparse.csc_matrix(matrix, dtype=np.float64)
            matrix.sum_duplicates()
            matrix.eliminate_zeros()
            matrix.sort_indices()
            finite = np.isfinite(matrix.data).all()
        else:
            matrix = np.asarray(matrix, dtype=np.float64)
            finite = np.isfinite(matrix).all()
        if matrix.ndim != 2 or matrix.shape[1] != len(names) or len(set(names)) != len(names):
            raise ValueError("Feature matrix and unique names must agree")
        if not finite:
            raise ValueError(
                "Candidate features must be finite; missingness is not silently imputed"
            )
        return matrix

    def fit(self, matrix, target, names: list[str]) -> TrainingScreen:
        matrix = self._validate(matrix, names)
        target = np.asarray(target)
        if target.ndim != 1 or len(target) != matrix.shape[0] or set(target.tolist()) != {0, 1}:
            raise ValueError("Training screening requires aligned binary labels from both classes")
        self.names_ = tuple(names)
        n, p = matrix.shape
        is_sparse = sparse.issparse(matrix)
        mean = np.asarray(matrix.mean(axis=0)).ravel()
        square = matrix.multiply(matrix) if is_sparse else matrix * matrix
        variance = np.maximum(np.asarray(square.mean(axis=0)).ravel() - mean * mean, 0)
        negative, positive = target == 0, target == 1
        m0 = np.asarray(matrix[negative].mean(axis=0)).ravel()
        m1 = np.asarray(matrix[positive].mean(axis=0)).ravel()
        between = negative.sum() * positive.sum() / n**2 * (m1 - m0) ** 2
        within = np.maximum(variance - between, 0)
        score = np.divide(between, within, out=np.zeros(p), where=within > 1e-12)
        reasons = np.full(p, "screen_budget", dtype=object)
        reasons[variance <= 1e-12] = "constant_or_near_constant"
        reasons[(variance > 1e-12) & (within <= 1e-12)] = "perfect_training_separator"
        hashes = {}
        for j in range(p):
            if reasons[j] != "screen_budget":
                continue
            if is_sparse:
                start, stop = matrix.indptr[j : j + 2]
                indices, data = matrix.indices[start:stop], matrix.data[start:stop]
                if len(data) < 3:
                    reasons[j] = "rare"
                    continue
                payload = indices.astype(np.int64).tobytes() + data.tobytes()
            else:
                data = matrix[:, j].copy()
                _, counts = np.unique(data, return_counts=True)
                if n - counts.max() < 3:
                    reasons[j] = "rare"
                    continue
                data[data == 0] = 0.0
                payload = np.ascontiguousarray(data).tobytes()
            key = hashlib.sha256(payload).hexdigest()
            if key in hashes:
                reasons[j] = "exact_duplicate"
            else:
                hashes[key] = j
        eligible = np.flatnonzero(reasons == "screen_budget")
        ranked = eligible[np.lexsort((np.asarray(names)[eligible], -score[eligible]))]
        selected = []
        limit = self.maximum * 4 if not is_sparse and self.correlation is not None else self.maximum
        standardized = []
        for j in ranked[:limit]:
            if len(selected) >= self.maximum:
                break
            if not is_sparse and self.correlation is not None:
                column = (matrix[:, j] - mean[j]) / np.sqrt(variance[j])
                if standardized:
                    correlations = np.abs(np.asarray(standardized) @ column / n)
                    if correlations.max() >= self.correlation:
                        reasons[j] = "high_training_correlation"
                        continue
                standardized.append(column)
            selected.append(int(j))
            reasons[j] = "retained"
        self.indices_ = np.asarray(selected, dtype=np.int64)
        self.catalog_ = pd.DataFrame(
            {
                "feature": names,
                "training_variance": variance,
                "training_effect_score": score,
                "decision": reasons,
            }
        )
        counts = dict(Counter(reasons.tolist()))
        self.audit_ = {
            "candidates": p,
            "retained": len(selected),
            "rejected": p - len(selected),
            "decisions": counts,
            "training_rows": n,
            "dtype": str(matrix.dtype),
            "missing_or_nonfinite": 0,
            "rare_minimum_rows": 3,
            "maximum": self.maximum,
            "correlation_threshold": None if is_sparse else self.correlation,
            "screening_scope": "training rows only; no validation labels or statistics",
            "catalog_sha256": hashlib.sha256(
                self.catalog_.to_csv(index=False).encode()
            ).hexdigest(),
        }
        if sum(counts.values()) != p:
            raise RuntimeError("Feature screening counts do not reconcile")
        return self

    def transform(self, matrix, names: list[str]):
        if not hasattr(self, "indices_"):
            raise ValueError("Fit the training screen before transforming")
        matrix = self._validate(matrix, names)
        if tuple(names) != self.names_:
            raise ValueError("Feature schema/order differs from the training screen")
        selected = matrix[:, self.indices_]
        return selected.tocsr() if sparse.issparse(selected) else selected
