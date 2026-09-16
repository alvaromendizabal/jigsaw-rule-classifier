"""Reference-only BM25 evidence; no target labels or neural encoding."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GroupKFold

from jigsaw_rules.data import normalize
from scripts.local_support_features import checked_labels
from scripts.multiprototype_features import validate_inputs

MEASURES = (
    "maximum",
    "top3",
    "top5",
    "mean",
    "dispersion",
    "overlap_fraction",
    "effective_coverage",
    "peak_excess",
)
NAMES = tuple(
    f"{family}/{label}/{measure}"
    for family in ("word_bm25", "character_bm25")
    for label in ("permitted", "violating", "margin")
    for measure in MEASURES
)
MODES = ("bm25", "tfidf", "no_length", "label_null")


def summarize(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError("finite nonempty evidence matrix required")
    ordered = np.sort(values, axis=1)
    top3 = ordered[:, -min(3, values.shape[1]) :].mean(1)
    top5 = ordered[:, -min(5, values.shape[1]) :].mean(1)
    total, squares = values.sum(1), (values**2).sum(1)
    effective = np.divide(
        total**2, squares * values.shape[1], out=np.zeros(len(values)), where=squares > 0
    )
    return np.column_stack(
        (
            values.max(1),
            top3,
            top5,
            values.mean(1),
            values.std(1),
            (values > 0).mean(1),
            effective,
            values.max(1) - top5,
        )
    )


def bm25_similarity(ref, query, lengths, *, b=0.75, k1=1.2):
    """Binary-query BM25, normalized by a query-only upper bound.

    IDF, mean analyzer length and document term counts come from references only.
    Query term repetition is not a fitted signal. Returned values lie in [0,1].
    """
    if not np.isfinite(b) or not 0 <= b <= 1 or not np.isfinite(k1) or k1 <= 0:
        raise ValueError("invalid fixed BM25 parameters")
    ref = ref.astype(float).tocsr(copy=True)
    query = query.astype(float).tocsr(copy=True)
    ref.sum_duplicates()
    query.sum_duplicates()
    ref.eliminate_zeros()
    query.eliminate_zeros()
    lengths = np.asarray(lengths, dtype=float)
    if (
        lengths.shape != (ref.shape[0],)
        or not len(lengths)
        or not np.isfinite(lengths).all()
        or (lengths < 0).any()
    ):
        raise ValueError("reference lengths must align")
    if ref.shape[1] != query.shape[1] or not np.isfinite(ref.data).all():
        raise ValueError("term matrices differ or are nonfinite")
    if (ref.data < 0).any() or (query.data < 0).any() or not np.isfinite(query.data).all():
        raise ValueError("term counts must be finite nonnegative")
    presence = ref.copy()
    presence.data[:] = 1
    df = np.asarray(presence.sum(0)).ravel()
    idf = np.log1p((len(lengths) - df + 0.5) / (df + 0.5))
    norm = k1 * (1 - b + b * lengths / max(float(lengths.mean()), 1e-12))
    rows = np.repeat(np.arange(len(lengths)), np.diff(ref.indptr))
    ref.data = (k1 + 1) * ref.data / (ref.data + norm[rows])
    query.data[:] = 1
    query = query.multiply(idf).tocsr()
    upper = np.asarray(query.sum(1)).ravel() * (k1 + 1)
    sim = (query @ ref.T).toarray()
    return np.clip(
        np.divide(sim, upper[:, None], out=np.zeros_like(sim), where=upper[:, None] > 0), 0, 1
    )


def feature_block(reference, labels, bodies, queries, query_bodies=None, *, seed=20260928):
    texts = np.asarray(list(bodies), dtype=object)
    qt = list(query_bodies) if query_bodies is not None else []
    if not qt or any(not isinstance(t, str) or not t.strip() for t in [*texts, *qt]):
        raise ValueError("nonempty reference/query text required")
    if any(len(t) > 50000 for t in [*texts, *qt]):
        raise ValueError("text exceeds the fixed CPU budget")
    y = checked_labels(labels, len(texts))
    order = np.argsort([normalize(t) for t in texts], kind="stable")
    texts, y = texts[order], y[order]
    label_null = np.random.default_rng(seed + 1).permutation(y)
    banks, diagnostics = {m: [] for m in MODES}, []
    for family in ("word", "character"):
        vectorizer = CountVectorizer(
            analyzer="word" if family == "word" else "char_wb",
            ngram_range=(1, 2) if family == "word" else (3, 5),
            strip_accents="unicode",
            min_df=1,
            max_features=6000,
        )
        lengths = np.asarray([len(vectorizer.build_analyzer()(t)) for t in texts], dtype=float)
        try:
            ref = vectorizer.fit_transform(texts)
            query = vectorizer.transform(qt)
            lexical = TfidfTransformer(sublinear_tf=True)
            rt = lexical.fit_transform(ref)
            tfidf = np.clip((lexical.transform(query) @ rt.T).toarray(), 0, 1)
            cosine = bm25_similarity(ref, query, lengths)
            no_length = bm25_similarity(ref, query, lengths, b=0)
            width = ref.shape[1]
            absent = float(np.mean(query.getnnz(axis=1) == 0))
        except ValueError as exc:
            if "empty vocabulary" not in str(exc):
                raise
            tfidf = cosine = no_length = np.zeros((len(qt), len(texts)))
            width, absent = 0, 1.0
        for mode in MODES:
            matrix = tfidf if mode == "tfidf" else no_length if mode == "no_length" else cosine
            target = label_null if mode == "label_null" else y
            classes = [summarize(matrix[:, target == cls]) for cls in (0, 1)]
            banks[mode].extend([*classes, classes[1] - classes[0]])
            diagnostics.append(
                {
                    "mode": mode,
                    "family": family,
                    "reference_rows": len(texts),
                    "vocabulary_columns": width,
                    "mean_reference_length": float(lengths.mean()),
                    "maximum_reference_length": float(lengths.max()),
                    "zero_query_fraction": absent,
                    "mean_similarity": float(matrix.mean()),
                    "query_used_for_vocabulary": False,
                }
            )
    return {m: np.column_stack(v) for m, v in banks.items()}, diagnostics


def cross_fitted(reference, labels, bodies, rules, queries, query_bodies, *, seed=20260928):
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
        raise ValueError("nonfinite BM25 evidence")
    return banks, diagnostics
