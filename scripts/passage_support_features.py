(
    "Localized lexical evidence; reference labels remain "
    # Keep this exact literal split for E501.
    "document labels, never passage labels."
    # Keep this exact literal split for E501.
)

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold

from jigsaw_rules.data import normalize
from scripts.local_support_features import checked_labels
from scripts.multiprototype_features import validate_inputs

MEASURES = (
    "maximum_evidence",
    "mean_evidence",
    "dispersion",
    "top2_evidence",
    "first_evidence",
    "last_evidence",
    "ending_shift",
    "strongest_position",
)
NAMES = tuple(
    f"{family}/{label}/{measure}"
    for family in ("word_passages", "character_passages")
    for label in ("permitted", "violating", "margin")
    for measure in MEASURES
)
MODES = ("passages", "whole_comment", "boundary_null", "label_null")


def passages(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("nonempty text required")
    if len(text) > 50000:
        raise ValueError("text exceeds preregistered CPU budget; no silent truncation")
    chunks = [v.strip() for v in re.split(r"(?<=[.!?;])\s+|\n+", text) if v.strip()]
    if len(chunks) > 12:
        chunks = [" ".join(group) for group in np.array_split(np.asarray(chunks), 12)]
    # Bound output count by merging adjacent chunks, never discarding tail evidence.
    if normalize(" ".join(chunks)) != normalize(text):
        raise AssertionError("passage decomposition changed the normalized content")
    return chunks


def boundary_control(text, count):
    words = re.findall(r"\S+", text)
    if count < 1 or count > len(words):
        raise ValueError("invalid fixed passage count")
    chunks = [" ".join(part) for part in np.array_split(np.asarray(words), count)]
    if normalize(" ".join(chunks)) != normalize(text):
        raise AssertionError("boundary control changed content")
    return chunks


def vectorizer_for(family):
    if family not in ("word", "character"):
        raise ValueError("unknown lexical family")
    return TfidfVectorizer(
        analyzer="word" if family == "word" else "char_wb",
        ngram_range=(1, 2) if family == "word" else (3, 5),
        min_df=1,
        max_features=6000,
        sublinear_tf=True,
        strip_accents="unicode",
    )


def aggregate_passages(evidence, counts):
    evidence = np.asarray(evidence, dtype=float)
    if evidence.ndim != 1 or sum(counts) != len(evidence) or not np.isfinite(evidence).all():
        raise ValueError("passage evidence/count alignment differs")
    if any(c < 1 or c > 12 for c in counts):
        raise ValueError("invalid passage count")
    rows, start = [], 0
    for count in counts:
        v = evidence[start : start + count]
        start += count
        position = float(np.argmax(v)) / max(1, count - 1)
        rows.append(
            [
                float(v.max()),
                float(v.mean()),
                float(v.std()),
                float(np.sort(v)[-min(2, count) :].mean()),
                float(v[0]),
                float(v[-1]),
                float(v[-1] - v[0]),
                position,
            ]
        )
    return np.asarray(rows)


def feature_block(reference, labels, bodies, queries, query_bodies=None, *, seed=20260927):
    # Cached vectors are deliberately not used by this new lexical feature family.
    # The unchanged anchor still uses them; shape/finiteness are checked by cross_fitted.
    texts = np.asarray(list(bodies), dtype=object)
    qt = list(query_bodies) if query_bodies is not None else []
    y = checked_labels(labels, len(texts))
    if not qt:
        raise ValueError("passage features require query text")
    order = np.argsort([normalize(t) for t in texts], kind="stable")
    texts, y = texts[order], y[order]
    if any(len(t) > 50000 for t in texts):
        raise ValueError("reference exceeds text budget")
    pieces = [passages(t) for t in qt]
    counts = [len(p) for p in pieces]
    controls = [boundary_control(t, n) for t, n in zip(qt, counts, strict=True)]
    groupings = {
        "passages": pieces,
        "whole_comment": [[t] for t in qt],
        "boundary_null": controls,
        "label_null": pieces,
    }
    target_null = np.random.default_rng(seed + 1).permutation(y)
    bank_pieces = {m: [] for m in MODES}
    diagnostics = []
    for family in ("word", "character"):
        encoder = vectorizer_for(family)
        try:
            ref = encoder.fit_transform(texts)
        except ValueError as exc:
            if "empty vocabulary" not in str(exc):
                raise
            encoder, ref = None, None
        # Transform three distinct decompositions once. Null labels reuse similarities.
        cache = {}
        zero_fraction = {}
        for mode in ("passages", "whole_comment", "boundary_null"):
            flat = [t for doc in groupings[mode] for t in doc]
            if encoder is None:
                sim = np.zeros((len(flat), len(texts)))
                zero_fraction[mode] = 1.0
            else:
                query = encoder.transform(flat)
                sim = (query @ ref.T).toarray()
                zero_fraction[mode] = float(np.mean(query.getnnz(axis=1) == 0))
            cache[mode] = np.clip(sim, 0, 1)
        for mode in MODES:
            source = "passages" if mode == "label_null" else mode
            sim = cache[source]
            target = target_null if mode == "label_null" else y
            classes = []
            for cls in (0, 1):
                values = sim[:, target == cls]
                k = min(5, values.shape[1])
                near = np.sort(values, axis=1)[:, -k:].mean(1)
                classes.append(aggregate_passages(near, [len(v) for v in groupings[mode]]))
            bank_pieces[mode].extend([*classes, classes[1] - classes[0]])
            diagnostics.append(
                {
                    "mode": mode,
                    "family": family,
                    "reference_rows": len(texts),
                    "vocabulary_columns": 0 if encoder is None else len(encoder.vocabulary_),
                    "mean_passages": float(np.mean([len(v) for v in groupings[mode]])),
                    "multiple_passage_fraction": float(np.mean(np.asarray(counts) > 1)),
                    "zero_lexical_fraction": zero_fraction[source],
                    "maximum_passages": max(len(v) for v in groupings[mode]),
                    "new_passage_labels": 0,
                    "query_used_for_vocabulary": False,
                }
            )
    return {m: np.column_stack(bank_pieces[m]) for m in MODES}, diagnostics


def cross_fitted(reference, labels, bodies, rules, queries, query_bodies, *, seed=20260927):
    r, y, texts, q, qt, groups = validate_inputs(
        reference, labels, bodies, rules, queries, query_bodies
    )
    train = {m: np.empty((len(r), 48)) for m in MODES}
    coverage, diagnostics = np.zeros(len(r), dtype=int), []
    for index, (keep, held) in enumerate(GroupKFold(3).split(r, groups=groups)):
        bank, stats = feature_block(r[keep], y[keep], texts[keep], r[held], texts[held], seed=seed)
        for mode in MODES:
            train[mode][held] = bank[mode]
        coverage[held] += 1
        diagnostics.extend({"inner_fold": str(index), "self_overlap": 0, **s} for s in stats)
    if not np.all(coverage == 1):
        raise ValueError("crossfit coverage differs")
    query, stats = feature_block(r, y, texts, q, qt, seed=seed)
    diagnostics.extend({"inner_fold": "outer_query", "self_overlap": 0, **s} for s in stats)
    banks = {m: (train[m], query[m]) for m in MODES}
    if not all(np.isfinite(a).all() for pair in banks.values() for a in pair):
        raise ValueError("nonfinite passage evidence features")
    return banks, diagnostics
