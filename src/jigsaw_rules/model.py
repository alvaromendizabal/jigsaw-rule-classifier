"""CPU reference models with explicit comment/example interactions."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from jigsaw_rules.data import EXAMPLES


class LexicalClassifier:
    """A transparent reference point; this model is not claimed to understand new policies."""

    def __init__(self, context: bool = True, seed: int = 2025):
        self.context = context
        self.seed = seed
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, max_features=40000, dtype=np.float64
        )
        self.classifier = LogisticRegression(
            C=2.0, solver="liblinear", max_iter=2000, random_state=seed
        )

    def _features(self, frame: pd.DataFrame):
        body = self.vectorizer.transform(frame.body)
        if not self.context:
            return body
        sims = [
            np.asarray(body.multiply(self.vectorizer.transform(frame[c])).sum(axis=1)).ravel()
            for c in ["rule", *EXAMPLES]
        ]
        pos = np.maximum(sims[1], sims[2])
        neg = np.maximum(sims[3], sims[4])
        extra = np.column_stack([*sims, pos, neg, pos - neg])
        return hstack([body, csr_matrix(extra)], format="csr")

    def fit(self, frame: pd.DataFrame) -> LexicalClassifier:
        columns = ["body", "rule", *EXAMPLES] if self.context else ["body"]
        corpus = [text for column in columns for text in frame[column]]
        self.vectorizer.fit(corpus)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            self.classifier.fit(self._features(frame), frame.rule_violation)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict_proba(self._features(frame))[:, 1]

    def coefficients(self) -> pd.DataFrame:
        names = self.vectorizer.get_feature_names_out().tolist()
        if self.context:
            names += [
                "similarity_rule",
                *[f"similarity_{c}" for c in EXAMPLES],
                "max_positive_similarity",
                "max_negative_similarity",
                "similarity_margin",
            ]
        return pd.DataFrame({"feature": names, "coefficient": self.classifier.coef_[0]})
