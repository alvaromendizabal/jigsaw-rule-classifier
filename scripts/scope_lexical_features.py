"""Occurrence-level lexical scope, with per-term energy-matched controls.

Scope flags are heuristic measurements, never moderation labels. Vocabulary,
IDF and diagnostic term weights are inherited from eligible training only.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.evidence_features import row_norms

TOKEN = re.compile(r"(?u)\b\w\w+\b")
STOP = re.compile(r"[.!?;\n]|\b(?:but|however|yet)\b", re.I)
NEGATOR = re.compile(
    r"\b(?:not|no|never|cannot|can['’]t|don['’]t|doesn['’]t|"
    # Explicit split survives formatter reflow.
    r"didn['’]t|isn['’]t|wasn['’]t|won['’]t|shouldn['’]t)\b",
    re.I,
)
CHANNELS = {
    "attribution": ("authored", "quoted", "code", "boundary"),
    "negation": ("unmarked", "negation_window", "boundary"),
}


def markup_mask(text):
    """Code wins over quote markup; single-quote prose remains unparsed."""
    mask = np.zeros(len(text), dtype=np.int8)
    for pattern in (r'"[^"\n]+"', r"“[^”\n]+”", r"(?m)^[ \t]*>[^\n]*"):
        for match in re.finditer(pattern, text):
            mask[match.start() : match.end()] = 1
    for pattern in (r"```[\s\S]*?```", r"`[^`\n]+`"):
        for match in re.finditer(pattern, text):
            mask[match.start() : match.end()] = 2
    return mask


def token_scopes(text, window=4):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("nonempty body text required")
    if isinstance(window, bool) or not isinstance(window, int) or window != 4:
        raise ValueError("the registered negation window is four tokens")
    text = text.lower()
    tokens = list(TOKEN.finditer(text))
    mask = markup_mask(text)
    attribution = np.asarray([mask[t.start()] for t in tokens], dtype=np.int8)
    cues = []
    for cue in NEGATOR.finditer(text):
        # Do not treat 'not only', 'not just', or 'no wonder' as ordinary negation.
        tail = text[cue.end() :]
        pseudo = cue.group() == "not" and re.match(r"\s+(?:only|just)\b", tail)
        pseudo = pseudo or (cue.group() == "no" and re.match(r"\s+wonder\b", tail))
        if not pseudo:
            cues.append(cue)
    negation = np.zeros(len(tokens), dtype=np.int8)
    last = None
    cursor = 0
    distance = 0
    for i, token in enumerate(tokens):
        while cursor < len(cues) and cues[cursor].end() <= token.start():
            last = cues[cursor]
            cursor += 1
            distance = 0
        if last is None:
            continue
        distance += 1
        if distance > window or mask[last.start()] != attribution[i]:
            continue
        # Scope cannot cross quote/code boundaries, punctuation, or an adversative.
        between = text[last.end() : token.start()]
        same_region = np.all(mask[last.end() : token.end()] == mask[last.start()])
        if not STOP.search(between) and same_region:
            negation[i] = 1
    return text, tokens, attribution, negation


def occurrence_counts(texts, vocabulary):
    """Keep original token adjacency; never join text after deleting a span."""
    rows, columns, totals = [], [], []
    scoped = {family: ([], [], []) for family in CHANNELS}
    coverage = []
    width = len(vocabulary)
    for row, body in enumerate(texts):
        text, tokens, attribution, negation = token_scopes(body)
        counts = Counter()
        per_family = {family: Counter() for family in CHANNELS}
        for i, token in enumerate(tokens):
            for size in (1, 2):
                if i + size > len(tokens):
                    continue
                word = " ".join(t.group() for t in tokens[i : i + size])
                if word not in vocabulary:
                    continue
                column = vocabulary[word]
                counts[column] += 1
                for family, labels in (("attribution", attribution), ("negation", negation)):
                    segment = labels[i : i + size]
                    boundary = size == 2 and (
                        segment[0] != segment[1]
                        or STOP.search(text[token.end() : tokens[i + 1].start()])
                    )
                    channel = len(CHANNELS[family]) - 1 if boundary else int(segment[0])
                    per_family[family][column + channel * width] += 1
        for column, count in counts.items():
            rows.append(row)
            columns.append(column)
            totals.append(count)
        for family, counts_by_channel in per_family.items():
            rr, cc, vv = scoped[family]
            for column, count in counts_by_channel.items():
                rr.append(row)
                cc.append(column)
                vv.append(count)
        coverage.append(
            {
                "tokens": len(tokens),
                "quoted_tokens": int((attribution == 1).sum()),
                "code_tokens": int((attribution == 2).sum()),
                "negation_tokens": int(negation.sum()),
                "unbalanced_double_quote": text.count('"') % 2,
                "empty_vocabulary_row": not bool(counts),
            }
        )
    shape = (len(texts), width)
    full = sparse.csr_matrix((totals, (rows, columns)), shape=shape, dtype=float)
    blocks = {}
    for family, (rr, cc, vv) in scoped.items():
        blocks[family] = sparse.csr_matrix(
            (vv, (rr, cc)), shape=(len(texts), width * len(CHANNELS[family])), dtype=float
        )
    return full, blocks, coverage


def energy_partition(values, total_counts, channel_counts, n_channels):
    """Allocate v*sqrt(scope_count/total), preserving term energy."""
    values = sparse.csr_matrix(values, dtype=float, copy=True)
    total = sparse.csr_matrix(total_counts, dtype=float, copy=True)
    part = sparse.csr_matrix(channel_counts, dtype=float, copy=True)
    if n_channels < 1 or part.shape != (values.shape[0], values.shape[1] * n_channels):
        raise ValueError("channel dimensions differ")
    if total.shape != values.shape:
        raise ValueError("total count dimensions differ")
    for matrix in (values, total, part):
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
            raise ValueError("finite nonnegative matrices required")
    d = values.shape[1]
    collapsed = sum(part[:, k * d : (k + 1) * d] for k in range(n_channels))
    difference = collapsed - total
    if np.max(np.abs(difference.data), initial=0.0) > 1e-12:
        raise ValueError("scope counts do not partition original occurrences")
    coo = part.tocoo()
    col = coo.col % d
    den = np.asarray(total[coo.row, col]).ravel() if coo.nnz else np.empty(0)
    if (den <= 0).any():
        raise ValueError("positive scope count has no original occurrence")
    amplitude = np.asarray(values[coo.row, col]).ravel() if coo.nnz else np.empty(0)
    distributed = sparse.csr_matrix(
        (amplitude * np.sqrt(coo.data / den), (coo.row, coo.col)), shape=part.shape
    )
    energy = sum(distributed[:, k * d : (k + 1) * d].power(2) for k in range(n_channels))
    diff = energy - values.power(2)
    error = float(np.max(np.abs(diff.data), initial=0.0))
    if error > 1e-10:
        raise ValueError("per-term energy parity failed")
    control = sparse.hstack(
        [values, sparse.csr_matrix((len(values.indptr) - 1, d * (n_channels - 1)))],
        format="csr",
    )
    return distributed, control, error


class ScopedLexicon:
    """Reuse the training-only vocabulary, IDF and weights."""

    def fit(self, train, spec, expected_names, evidence):
        text = ("RULE: " + train.rule + "\nBODY: " + train.body).tolist()
        self.vectorizer_ = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=spec["word_features"], sublinear_tf=True
        ).fit(text)
        if not np.array_equal(self.vectorizer_.get_feature_names_out(), expected_names):
            raise ValueError("inherited word vocabulary differs")
        self.evidence_ = evidence
        return self

    def transform(self, frame):
        if not hasattr(self, "vectorizer_"):
            raise ValueError("fit using eligible training first")
        if set(frame.columns) != {"body", "rule"}:
            raise ValueError("scope feature frame must contain body and rule only")
        texts = frame.body.tolist()
        total, counts, coverage = occurrence_counts(texts, self.vectorizer_.vocabulary_)
        value = total.copy()
        value.data = 1.0 + np.log(value.data)
        value = value.multiply(self.vectorizer_.idf_).tocsr()
        norms = row_norms(value)
        scale = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
        value = value.multiply(scale[:, None]).tocsr()
        expected = self.vectorizer_.transform(texts)
        diff = value - expected
        if np.max(np.abs(diff.data), initial=0.0) > 1e-12:
            raise ValueError("body word reconstruction parity failed")
        value, _ = self.evidence_.transform(value, frame.rule, mode="rule")
        blocks, checks = {}, []
        for family, names in CHANNELS.items():
            scoped, control, error = energy_partition(value, total, counts[family], len(names))
            blocks[family] = {"scope": scoped, "copy": control}
            checks.append(
                {
                    "family": family,
                    "columns": scoped.shape[1],
                    "scoped_nonzeros": scoped.nnz,
                    "copy_nonzeros": control.nnz,
                    "max_per_term_energy_error": error,
                }
            )
        return blocks, coverage, checks
