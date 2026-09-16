(
    "Reference consistency is a feature, never authority to "
    # Keep this exact literal split for E501.
    "change a supplied label."
    # Keep this exact literal split for E501.
)

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.model_selection import GroupKFold

from jigsaw_rules.data import normalize
from scripts.local_support_features import checked_labels, unit
from scripts.multiprototype_features import validate_inputs

MEASURES = (
    "affinity_change",
    "similarity_change",
    "active_quality",
    "quality_dispersion",
    "high_quality_mass",
    "effective_fraction_change",
    "top5_quality",
    "quality_similarity_cov",
)
NAMES = tuple(
    f"{family}/{label}/{measure}"
    for family in ("agreement", "reciprocity")
    for label in ("permitted", "violating", "margin")
    for measure in MEASURES
)
MODES = ("observed", "quality_null", "geometry_only", "label_null")


def reference_quality(reference, labels):
    (
        "Leave-one-reference-out kNN statistics, recomputed "
        # Keep this exact literal split for E501.
        "inside each crossfit pool."
        # Keep this exact literal split for E501.
    )
    r = unit(reference)
    y = checked_labels(labels, len(r))
    if len(r) < 3:
        raise ValueError("at least three references required")
    k = min(7, len(r) - 1)
    similarity = np.clip(r @ r.T, -1, 1)
    np.fill_diagonal(similarity, -np.inf)
    neighbors = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
    if any(i in row for i, row in enumerate(neighbors)):
        raise AssertionError("reference self-neighbor escaped exclusion")
    graph = np.zeros((len(r), len(r)), dtype=bool)
    graph[np.arange(len(r))[:, None], neighbors] = True
    mutual = graph & graph.T
    same = y[:, None] == y[None, :]
    ordinary = (1 + (same & graph).sum(1)) / (2 + k)
    reciprocal = (1 + (same & mutual).sum(1)) / (2 + mutual.sum(1))
    centrality = (1 + mutual.sum(1)) / (2 + k)
    # Never remove a reference or modify its label, even when its neighbors disagree.
    qualities = np.column_stack([0.25 + 0.75 * ordinary, 0.25 + 0.75 * reciprocal])
    return (
        qualities,
        0.25 + 0.75 * centrality,
        {
            "neighbors_per_reference": k,
            "mean_neighbor_agreement": float(ordinary.mean()),
            "mean_reciprocal_agreement": float(reciprocal.mean()),
            "mean_reciprocal_degree": float(mutual.sum(1).mean()),
            "self_neighbors": 0,
        },
    )


def class_statistics(similarity, quality):
    s = np.asarray(similarity, dtype=float)
    quality = np.asarray(quality, dtype=float)
    if s.ndim != 2 or quality.shape != (s.shape[1],) or not np.isfinite(s).all():
        raise ValueError("finite, aligned similarity required")
    if not np.isfinite(quality).all() or ((quality < 0.25) | (quality > 1)).any():
        raise ValueError("reference quality must lie in [0.25,1]")
    logits = s / 0.1
    p = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    weighted = p * quality
    weighted /= weighted.sum(1, keepdims=True)
    mean_quality = p @ quality
    centered_s = s - (p * s).sum(1, keepdims=True)
    centered_q = quality[None, :] - mean_quality[:, None]
    top = np.argsort(-s, axis=1, kind="stable")[:, : min(5, s.shape[1])]
    affinity_change = 0.1 * (
        logsumexp(logits + np.log(quality), axis=1)
        - np.log(quality.sum())
        - logsumexp(logits, axis=1)
        + np.log(len(quality))
    )
    return np.column_stack(
        [
            affinity_change,
            ((weighted - p) * s).sum(1),
            mean_quality,
            np.sqrt(np.maximum((p * centered_q**2).sum(1), 0)),
            p @ (quality >= 0.75),
            (1 / (weighted**2).sum(1) - 1 / (p**2).sum(1)) / len(quality),
            quality[top].mean(1),
            (p * centered_s * centered_q).sum(1),
        ]
    )


def feature_block(reference, labels, bodies, queries, query_bodies=None, *, seed=20260926):
    r, q = unit(reference), unit(queries)
    y = checked_labels(labels, len(r))
    order = np.argsort([normalize(t) for t in bodies], kind="stable")
    r, y = r[order], y[order]
    if r.shape[1] != q.shape[1]:
        raise ValueError("reference/query dimensions differ")
    shuffled = np.random.default_rng(seed + 1).permutation(y)
    observed, centrality, observed_stats = reference_quality(r, y)
    null, _, null_stats = reference_quality(r, shuffled)
    quality_null = observed.copy()
    for cls in (0, 1):
        where = np.flatnonzero(y == cls)
        quality_null[where] = observed[np.random.default_rng(seed + 10 + cls).permutation(where)]
    sim = np.clip(q @ r.T, -1, 1)
    banks, diagnostics = {}, []
    for mode in MODES:
        target = shuffled if mode == "label_null" else y
        qualities = {
            "observed": observed,
            "quality_null": quality_null,
            "geometry_only": np.column_stack([centrality, centrality]),
            "label_null": null,
        }[mode]
        families = []
        for family in range(2):
            classes = [
                class_statistics(sim[:, target == c], qualities[target == c, family])
                for c in (0, 1)
            ]
            families.extend([*classes, classes[1] - classes[0]])
        banks[mode] = np.column_stack(families)
        stats = null_stats if mode == "label_null" else observed_stats
        for cls in (0, 1):
            w = qualities[target == cls]
            diagnostics.append(
                {
                    "mode": mode,
                    "class": cls,
                    "reference_rows": int((target == cls).sum()),
                    "mean_quality": float(w.mean()),
                    "minimum_quality": float(w.min()),
                    "maximum_quality": float(w.max()),
                    "labels_changed": False,
                    "query_used_for_quality": False,
                    **stats,
                }
            )
    return banks, diagnostics


def cross_fitted(reference, labels, bodies, rules, queries, query_bodies, *, seed=20260926):
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
        raise ValueError("nonfinite reference-consistency features")
    return banks, diagnostics
