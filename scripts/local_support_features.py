"""Same-rule density evidence on cached vectors; no encoder or query-label access."""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logsumexp
from sklearn.model_selection import GroupKFold

from jigsaw_rules.data import normalize

BASIC = tuple(
    f"basic/{label}/{stat}"
    for label in ("permitted", "violating", "margin")
    for stat in ("centroid", "nearest", "top5")
)
LOCAL = (
    "local/log_density_permitted",
    "local/log_density_violating",
    "local/log_density_ratio",
    "local/evidence_share",
    "local/entropy",
    "local/nearest_permitted",
    "local/nearest_violating",
    "local/nearest_margin",
    "local/log_effective_permitted",
    "local/log_effective_violating",
    "local/effective_margin",
    "local/log_relative_radius",
)
NAMES = (*BASIC, *LOCAL)


def unit(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("nonempty finite 2D vectors required")
    if len(x) > 4000 or x.shape[1] > 4096:
        raise ValueError("declared vector memory budget exceeded")
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if (norm < 1e-12).any():
        raise ValueError("zero-norm vector")
    return x / norm


def checked_labels(labels, size):
    y = np.asarray(labels)
    if y.shape != (size,) or not np.isin(y, [0, 1]).all():
        raise ValueError("aligned binary labels required")
    if min(np.count_nonzero(y == c) for c in (0, 1)) < 2:
        raise ValueError("at least two training references per class required")
    return y.astype(int)


def geometry(reference, labels, queries, *, neighbors=10, shuffle_seed=None):
    """21 fixed features, using one policy's explicitly labeled reference pool.

    Radii are label-independent kth-neighbor distances. No query-to-query
    operation or fitted query statistic is used. Shuffling touches reference
    labels only; it never receives evaluation-query targets.
    """
    r, q = unit(reference), unit(queries)
    if r.shape[1] != q.shape[1]:
        raise ValueError("reference/query dimensions differ")
    y = checked_labels(labels, len(r)).copy()
    if not isinstance(neighbors, int) or not 1 <= neighbors <= 25:
        raise ValueError("neighbors must be in [1,25]")
    if shuffle_seed is not None:
        y = np.random.default_rng(shuffle_seed).permutation(y)
    rr = np.clip(2 * (1 - r @ r.T), 0, 4)
    np.fill_diagonal(rr, np.inf)
    k = min(neighbors, len(r) - 1)
    radius = np.sqrt(np.partition(rr, k - 1, axis=1)[:, k - 1])
    radius = np.maximum(radius, 1e-6)
    cos = np.clip(q @ r.T, -1, 1)
    distance2 = np.clip(2 * (1 - cos), 0, 4)
    query_radius = np.sqrt(np.partition(distance2, k - 1, axis=1)[:, k - 1])
    query_radius = np.maximum(query_radius, 1e-6)
    kernel = -distance2 / (query_radius[:, None] * radius[None, :])
    basic, density, nearest, effective = [], [], [], []
    for cls in (0, 1):
        keep = y == cls
        center = r[keep].mean(axis=0)
        center_norm = np.linalg.norm(center)
        if center_norm < 1e-12:
            raise ValueError("degenerate class centroid")
        similarities = cos[:, keep]
        top = min(5, int(keep.sum()))
        basic.extend(
            [
                q @ (center / center_norm),
                similarities.max(axis=1),
                np.partition(similarities, -top, axis=1)[:, -top:].mean(axis=1),
            ]
        )
        values = kernel[:, keep]
        total = logsumexp(values, axis=1)
        density.append(total - np.log(keep.sum()))
        nearest.append(values.max(axis=1))
        # Effective sample count / class count; stable in log space.
        ess = 2 * total - logsumexp(2 * values, axis=1) - np.log(keep.sum())
        effective.append(ess)
    basic = np.column_stack(basic)
    basic = np.column_stack([basic, basic[:, 3:] - basic[:, :3]])
    ratio = density[1] - density[0]
    posterior = expit(ratio)
    clipped = np.clip(posterior, 1e-12, 1 - 1e-12)
    entropy = -clipped * np.log(clipped) - (1 - clipped) * np.log1p(-clipped)
    local = np.column_stack(
        [
            *density,
            ratio,
            posterior,
            entropy,
            *nearest,
            nearest[1] - nearest[0],
            *effective,
            effective[1] - effective[0],
            np.log(query_radius / np.median(radius)),
        ]
    )
    out = np.column_stack([basic, local])
    if out.shape != (len(q), len(NAMES)) or not np.isfinite(out).all():
        raise ValueError("nonfinite or misaligned geometry features")
    return out


def cross_fitted(reference, labels, bodies, queries, *, neighbors=10, seed=None):
    """Exclude each entire normalized-text group from its own feature references.

    The representation matrix itself is held fixed. In the adapted-vector
    sensitivity arm, this is NOT an out-of-fold encoder.
    """
    r = unit(reference)
    y = checked_labels(labels, len(r))
    texts = list(bodies)
    if len(texts) != len(r) or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("aligned nonempty text groups required")
    groups = np.asarray([normalize(t) for t in texts])
    if len(np.unique(groups)) < 3:
        raise ValueError("at least three text groups required")
    train_features = np.empty((len(r), len(NAMES)))
    coverage = np.zeros(len(r), dtype=int)
    metadata = []
    for index, (keep, held) in enumerate(GroupKFold(n_splits=3).split(r, groups=groups)):
        if set(groups[keep]) & set(groups[held]):
            raise AssertionError("self-text entered reference pool")
        local_seed = None if seed is None else seed + index
        train_features[held] = geometry(
            r[keep],
            y[keep],
            r[held],
            neighbors=neighbors,
            shuffle_seed=local_seed,
        )
        coverage[held] += 1
        metadata.append(
            {
                "inner_fold": index,
                "references": len(keep),
                "held_rows": len(held),
                "self_text_overlap": 0,
                "permitted_references": int((y[keep] == 0).sum()),
                "violating_references": int((y[keep] == 1).sum()),
            }
        )
    if not np.all(coverage == 1):
        raise AssertionError("cross-fitting did not cover each training row exactly once")
    evaluation = geometry(
        r,
        y,
        queries,
        neighbors=neighbors,
        shuffle_seed=None if seed is None else seed + 3,
    )
    return train_features, evaluation, metadata
