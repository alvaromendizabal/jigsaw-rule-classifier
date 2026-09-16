"""Same-rule, cross-fitted opposing-exemplar contrasts on frozen cached vectors."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
from sklearn.model_selection import GroupKFold

from jigsaw_rules.data import normalize
from scripts.local_support_features import checked_labels, unit

GLOBAL = (
    "pair/global_mean",
    "pair/global_std",
    "pair/global_soft_vote",
    "pair/global_positive_fraction",
)
LOCAL = (
    "pair/nearest_midpoint_margin",
    "pair/top5_mean",
    "pair/top20_mean",
    "pair/top20_std",
    "pair/weighted_mean",
    "pair/weighted_std",
    "pair/weighted_soft_vote",
    "pair/local_minus_global",
)
NAMES = (*GLOBAL, *LOCAL)
MODES = ("matched", "repaired", "orientation", "unnormalized")


def canonical_reference(reference, labels, bodies):
    r = unit(reference)
    y = checked_labels(labels, len(r))
    text = list(bodies)
    if len(text) != len(r) or any(not isinstance(t, str) or not t.strip() for t in text):
        raise ValueError("aligned nonempty reference text required")
    groups = np.asarray([normalize(t) for t in text])
    if len(np.unique(groups)) != len(groups):
        raise ValueError("duplicate reference texts must be resolved before matching")
    order = np.argsort(groups, kind="stable")
    return r[order], y[order], groups[order]


def matched_endpoints(reference, labels, bodies):
    """One-to-one maximum-cosine assignment, not repeated nearest-neighbor hubs.

    Both endpoints have explicit, opposite labels under the same policy. The
    larger class can have unused examples; comparisons retain identical endpoints.
    Matching and all tie ordering are functions of reference inputs only.
    """
    r, y, texts = canonical_reference(reference, labels, bodies)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    cost = -np.clip(r[pos] @ r[neg].T, -1, 1)
    a, b = linear_sum_assignment(cost)
    p, n = pos[a], neg[b]
    if len(set(p)) != len(p) or len(set(n)) != len(n):
        raise AssertionError("matching reused an endpoint")
    # Canonical order keeps null controls stable under a reference-row permutation.
    order = sorted(range(len(p)), key=lambda i: (texts[p[i]], texts[n[i]]))
    p, n = p[order], n[order]
    metadata = {
        "reference_rows": len(r),
        "permitted_references": len(neg),
        "violating_references": len(pos),
        "pairs": len(p),
        "unused_reference_rows": len(r) - 2 * len(p),
        "maximum_endpoint_reuse": 1,
    }
    return r[p], r[n], metadata


def summarize_pairs(positive, negative, queries, *, normalize_margin=True, signs=None):
    q = unit(queries)
    if positive.shape != negative.shape or q.shape[1] != positive.shape[1]:
        raise ValueError("pair/query dimensions differ")
    diff = positive - negative
    mid = positive + negative
    gaps = np.linalg.norm(diff, axis=1)
    midnorm = np.linalg.norm(mid, axis=1)
    usable = (gaps > 1e-6) & (midnorm > 1e-6)
    if not usable.any():
        raise ValueError("no nondegenerate opposing pairs")
    diff, mid, gaps, midnorm = diff[usable], mid[usable], gaps[usable], midnorm[usable]
    directions = diff / gaps[:, None] if normalize_margin else diff
    if signs is not None:
        s = np.asarray(signs)
        if s.shape != (len(positive),) or not np.isin(s, [-1, 1]).all():
            raise ValueError("invalid orientation-control signs")
        directions = directions * s[usable, None]
    proximity = np.clip(q @ (mid / midnorm[:, None]).T, -1, 1)
    margin = q @ directions.T
    # A fixed temperature, not validation-selected. Never estimate it on queries.
    weights = softmax(proximity / 0.1, axis=1)
    ranking = np.argsort(-proximity, axis=1, kind="stable")
    ordered = np.take_along_axis(margin, ranking, axis=1)
    top5, top20 = ordered[:, :5], ordered[:, :20]
    mean = margin.mean(axis=1)
    weighted = (weights * margin).sum(axis=1)
    variance = (weights * (margin - weighted[:, None]) ** 2).sum(axis=1)
    votes = np.tanh(5 * margin)
    values = np.column_stack(
        [
            mean,
            margin.std(axis=1),
            votes.mean(axis=1),
            (margin > 0).mean(axis=1),
            ordered[:, 0],
            top5.mean(axis=1),
            top20.mean(axis=1),
            top20.std(axis=1),
            weighted,
            np.sqrt(np.maximum(variance, 0)),
            (weights * votes).sum(axis=1),
            weighted - mean,
        ]
    )
    if values.shape != (len(q), len(NAMES)) or not np.isfinite(values).all():
        raise ValueError("nonfinite or misaligned pair features")
    stats = {
        "usable_pairs": int(usable.sum()),
        "degenerate_pairs_excluded": int((~usable).sum()),
        "mean_pair_cosine": float(np.mean(np.sum(positive * negative, axis=1))),
        "median_separation": float(np.median(gaps)),
        "min_separation": float(gaps.min()),
    }
    return values, stats


def paired_geometry(reference, labels, bodies, queries, *, seed):
    p, n, meta = matched_endpoints(reference, labels, bodies)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(n))
    signs = rng.choice([-1, 1], size=len(p))
    outputs, stats = {}, {}
    for mode in MODES:
        other = n[permutation] if mode == "repaired" else n
        outputs[mode], record = summarize_pairs(
            p,
            other,
            queries,
            normalize_margin=(mode != "unnormalized"),
            signs=signs if mode == "orientation" else None,
        )
        stats[mode] = {**meta, **record, "same_selected_endpoints": True}
    return outputs, stats


def cross_fitted(reference, labels, bodies, rules, queries, *, seed):
    r = unit(reference)
    y = checked_labels(labels, len(r))
    texts, policies = list(bodies), list(rules)
    if len(texts) != len(r) or len(policies) != len(r):
        raise ValueError("reference metadata misaligned")
    if any(not isinstance(v, str) or not v.strip() for v in texts + policies):
        raise ValueError("reference metadata must be nonempty text")
    if len({normalize(v) for v in policies}) != 1:
        raise ValueError("opposing references must belong to the same rule")
    groups = np.asarray([normalize(t) for t in texts])
    if len(np.unique(groups)) != len(r) or len(r) < 6:
        raise ValueError("unique normalized reference texts and six rows required")
    training = {m: np.empty((len(r), len(NAMES))) for m in MODES}
    coverage = np.zeros(len(r), dtype=int)
    metadata = []
    for index, (keep, held) in enumerate(GroupKFold(n_splits=3).split(r, groups=groups)):
        if set(groups[keep]) & set(groups[held]):
            raise AssertionError("self text entered the reference pool")
        result, info = paired_geometry(r[keep], y[keep], groups[keep], r[held], seed=seed + index)
        for mode in MODES:
            training[mode][held] = result[mode]
            metadata.append(
                {"inner_fold": index, "mode": mode, "self_text_overlap": 0, **info[mode]}
            )
        coverage[held] += 1
    if not np.all(coverage == 1):
        raise AssertionError("training rows not cross-fitted exactly once")
    evaluation, info = paired_geometry(r, y, groups, queries, seed=seed + 3)
    metadata.extend({"inner_fold": "outer_query", "mode": m, **info[m]} for m in MODES)
    return {m: (training[m], evaluation[m]) for m in MODES}, metadata
