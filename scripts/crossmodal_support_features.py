(
    "Cross-fitted lexical/semantic agreement using explicitly l"
    # Keep long literals split for E501.
    "abeled same-rule supports."
    # Keep long literals split for E501.
)

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold

from jigsaw_rules.data import normalize
from scripts.local_support_features import checked_labels, unit

MEASURES = (
    "lexical_nearest",
    "lexical_top5",
    "semantic_at_lexical_top5",
    "lexical_at_semantic_top5",
    "top5_overlap",
    "joint_top5",
    "affinity_weighted_lexical",
    "alignment_covariance",
)
NAMES = tuple(
    f"{family}/{label}/{measure}"
    for family in ("word", "character")
    for label in ("permitted", "violating", "margin")
    for measure in MEASURES
)
MODES = ("aligned", "alignment_null", "label_null", "lexical_only")


def validate_inputs(reference, labels, bodies, rules, queries, query_bodies):
    r, q = unit(reference), unit(queries)
    y = checked_labels(labels, len(r))
    texts, rules, qt = list(bodies), list(rules), list(query_bodies)
    if r.shape[1] != q.shape[1] or len(q) < 1:
        raise ValueError("query dimension/length mismatch")
    if len(texts) != len(r) or len(rules) != len(r) or len(qt) != len(q):
        raise ValueError("reference/query metadata alignment mismatch")
    if any(not isinstance(t, str) or not t.strip() for t in texts + rules + qt):
        raise ValueError("nonempty string text/rule required")
    groups = np.asarray([normalize(t) for t in texts])
    if len(np.unique(groups)) != len(groups):
        raise ValueError("unique normalized reference texts required")
    if len(set(map(normalize, rules))) != 1:
        raise ValueError("same rule references required")
    if set(groups) & set(map(normalize, qt)):
        raise ValueError("query text overlaps reference pool")
    if len(r) < 9:
        raise ValueError("at least nine reference rows required for cross-fitting")
    return r, y, np.asarray(texts), q, np.asarray(qt), groups


def lexical_cosine(texts, query_texts, family):
    if family not in ("word", "character"):
        raise ValueError("unknown lexical family")
    kwargs = (
        {"analyzer": "word", "ngram_range": (1, 2)}
        if family == "word"
        else {"analyzer": "char_wb", "ngram_range": (3, 5)}
    )
    vectorizer = TfidfVectorizer(
        **kwargs, min_df=1, max_features=6000, sublinear_tf=True, norm="l2"
    )
    try:
        rx = vectorizer.fit_transform(texts)
    except ValueError as exc:
        if "empty vocabulary" not in str(exc):
            raise
        return np.zeros((len(query_texts), len(texts))), 0, 1.0
    qx = vectorizer.transform(query_texts)
    return (
        np.clip((qx @ rx.T).toarray(), 0.0, 1.0),
        len(vectorizer.vocabulary_),
        float(np.mean(qx.getnnz(axis=1) == 0)),
    )


def summarize(lexical, semantic, labels):
    lex, sem = np.asarray(lexical, float), np.asarray(semantic, float)
    y = checked_labels(labels, lex.shape[1])
    if lex.ndim != 2 or lex.shape != sem.shape or not np.isfinite(lex + sem).all():
        raise ValueError("finite aligned similarity matrices required")
    blocks = []
    for cls in (0, 1):
        a, b = lex[:, y == cls], sem[:, y == cls]
        k = min(5, a.shape[1])
        ia = np.argsort(-a, axis=1, kind="stable")[:, :k]
        ib = np.argsort(-b, axis=1, kind="stable")[:, :k]
        rows = np.arange(len(a))[:, None]
        active = a[rows, ia] > 0
        active_count = np.maximum(active.sum(1), 1)
        overlap = ((ia[:, :, None] == ib[:, None, :]).any(axis=2) & active).sum(1) / active_count
        semantic_at_lexical = (b[rows, ia] * active).sum(1) / active_count
        joint = a * np.clip(b, 0, 1)
        top_joint = np.partition(joint, -k, axis=1)[:, -k:].mean(axis=1)
        weights = np.exp((b - b.max(axis=1, keepdims=True)) / 0.1)
        weighted_lex = (weights * a).sum(1) / weights.sum(1)
        blocks.append(
            np.column_stack(
                [
                    a.max(1),
                    a[rows, ia].mean(1),
                    semantic_at_lexical,
                    a[rows, ib].mean(1),
                    overlap,
                    top_joint,
                    weighted_lex,
                    (a * b).mean(1) - a.mean(1) * b.mean(1),
                ]
            )
        )
    return np.column_stack([*blocks, blocks[1] - blocks[0]])


def feature_block(reference, labels, bodies, queries, query_bodies, *, seed=20260922):
    r, q = unit(reference), unit(queries)
    y = checked_labels(labels, len(r))
    texts = np.asarray(list(bodies))
    order = np.argsort([normalize(t) for t in texts], kind="stable")
    r, y, texts = r[order], y[order], texts[order]
    if r.shape[1] != q.shape[1]:
        raise ValueError("reference/query dimensions differ")
    sem = np.clip(q @ r.T, -1, 1)
    permutation = np.arange(len(r))
    rng = np.random.default_rng(seed)
    for cls in (0, 1):
        indices = np.flatnonzero(y == cls)
        permutation[indices] = rng.permutation(indices)
    shuffled_y = np.random.default_rng(seed + 1).permutation(y)
    banks = {m: [] for m in MODES}
    diagnostics = []
    for family in ("word", "character"):
        lex, width, missing = lexical_cosine(texts, query_bodies, family)
        aligned = summarize(lex, sem, y)
        banks["aligned"].append(aligned)
        banks["alignment_null"].append(summarize(lex, sem[:, permutation], y))
        banks["label_null"].append(summarize(lex, sem, shuffled_y))
        only = aligned.copy()
        only[:, [j for j in range(24) if j % 8 >= 2]] = 0
        banks["lexical_only"].append(only)
        diagnostics.append(
            {
                "family": family,
                "vocabulary_columns": width,
                "zero_lexical_query_fraction": missing,
                "reference_rows": len(r),
                "query_rows": len(q),
                "mean_top5_agreement": float(aligned[:, [4, 12]].mean()),
                "mean_lexical_nearest": float(aligned[:, [0, 8]].mean()),
            }
        )
    return {m: np.column_stack(v) for m, v in banks.items()}, diagnostics


def cross_fitted(reference, labels, bodies, rules, queries, query_bodies, *, seed=20260922):
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
    if not all(np.isfinite(a).all() for v in banks.values() for a in v):
        raise ValueError("nonfinite crossmodal feature bank")
    return banks, diagnostics
