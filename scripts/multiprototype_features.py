(
    "Fixed-capacity class subtypes fitted exclusively on same-r"
    # Keep long literals split for E501.
    "ule references."
    # Keep long literals split for E501.
)

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.model_selection import GroupKFold

from jigsaw_rules.data import normalize
from scripts.local_support_features import checked_labels, unit

LOCATION = ("nearest", "top2", "mean", "soft_affinity", "nearest_gap", "between_mode_sd")
COVERAGE = (
    "best_scaled_affinity",
    "mean_scaled_affinity",
    "mode_entropy",
    "effective_modes",
    "near_mode_coverage",
    "nearest_mode_mass",
)
NAMES = tuple(
    f"{family}/{label}/{measure}"
    for family, measures in (("location", LOCATION), ("coverage", COVERAGE))
    for label in ("permitted", "violating", "margin")
    for measure in measures
)
MODES = ("clustered", "random_partition", "single_center", "label_null")


def validate_inputs(reference, labels, bodies, rules, queries, query_bodies):
    r, q = unit(reference), unit(queries)
    y = checked_labels(labels, len(r))
    texts, rules, qt = list(bodies), list(rules), list(query_bodies)
    if r.shape[1] != q.shape[1] or not len(q):
        raise ValueError("query dimension/length mismatch")
    if len(texts) != len(r) or len(rules) != len(r) or len(qt) != len(q):
        raise ValueError("reference/query metadata alignment mismatch")
    if any(not isinstance(t, str) or not t.strip() for t in texts + rules + qt):
        raise ValueError("nonempty text/rule required")
    groups = np.asarray([normalize(t) for t in texts])
    if len(np.unique(groups)) != len(groups):
        raise ValueError("unique reference text required")
    if len(set(map(normalize, rules))) != 1:
        raise ValueError("same rule references required")
    if set(groups) & set(map(normalize, qt)):
        raise ValueError("query text overlaps reference pool")
    if len(r) < 9:
        raise ValueError("at least nine reference rows required")
    return r, y, np.asarray(texts), q, np.asarray(qt), groups


def spherical_partition(values, max_modes=4, iterations=12):
    (
        "Deterministic farthest-first initialization, then a fixed "
        # Keep long literals split for E501.
        "spherical Lloyd budget."
        # Keep long literals split for E501.
    )
    x = unit(values)
    if max_modes != 4 or iterations != 12:
        raise ValueError("registered clustering budget differs")
    # Do not create many prototypes from very small class pools.
    k = min(4, max(1, len(x) // 4))
    center = x.mean(0)
    if np.linalg.norm(center) <= 1e-12:
        raise ValueError("degenerate class mean")
    first = int(np.argmax(x @ (center / np.linalg.norm(center))))
    seeds = [first]
    while len(seeds) < k:
        closeness = (x @ x[seeds].T).max(1)
        closeness[seeds] = np.inf
        candidate = int(np.argmin(closeness))
        if closeness[candidate] > 1 - 1e-10:
            break
        seeds.append(candidate)
    prototypes = x[seeds].copy()
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iterations):
        labels = np.argmax(x @ prototypes.T, axis=1)
        occupied = np.unique(labels)
        prototypes = np.stack([x[labels == j].mean(0) for j in occupied])
        if (np.linalg.norm(prototypes, axis=1) <= 1e-12).any():
            raise ValueError("degenerate cluster mean")
        prototypes = unit(prototypes)
    # The returned partition defines the actual summaries, including collapsed modes.
    labels = np.argmax(x @ prototypes.T, axis=1)
    return np.searchsorted(np.unique(labels), labels)


def prototypes(values, assignment):
    x = unit(values)
    labels = np.asarray(assignment)
    if labels.shape != (len(x),) or labels.dtype.kind not in "iu":
        raise ValueError("aligned integer assignments required")
    centers, counts, radii = [], [], []
    for index in np.unique(labels):
        rows = x[labels == index]
        mean = rows.mean(0)
        if np.linalg.norm(mean) <= 1e-12:
            raise ValueError("degenerate prototype")
        mean /= np.linalg.norm(mean)
        centers.append(mean)
        counts.append(len(rows))
        # A fixed floor prevents near-singleton clusters from extreme extrapolation.
        radii.append(max(float((1 - rows @ mean).mean()), 0.05))
    return np.asarray(centers), np.asarray(counts) / len(x), np.asarray(radii)


def class_features(query, model):
    q = unit(query)
    centers, mass, radius = model
    similarity = np.clip(q @ centers.T, -1, 1)
    n = similarity.shape[1]
    ordered = np.sort(similarity, axis=1)
    best = ordered[:, -1]
    top2 = ordered[:, -min(n, 2) :].mean(1)
    gap = best - ordered[:, -2] if n > 1 else np.zeros(len(q))
    affinity = 0.1 * logsumexp(similarity / 0.1 + np.log(mass), axis=1)
    location = np.column_stack(
        [
            best,
            top2,
            similarity @ mass,
            affinity,
            gap,
            np.sqrt(np.maximum((similarity**2) @ mass - (similarity @ mass) ** 2, 0)),
        ]
    )
    scaled = np.exp(np.clip(-(1 - similarity) / radius, -80, 0))
    response = scaled * mass
    response /= response.sum(1, keepdims=True)
    entropy = -(response * np.log(np.maximum(response, 1e-30))).sum(1)
    coverage = np.column_stack(
        [
            scaled.max(1),
            scaled @ mass,
            entropy / np.log(max(2, n)),
            np.exp(entropy) / n,
            ((1 - similarity) <= radius) @ mass,
            mass[np.argmax(similarity, axis=1)],
        ]
    )
    return location, coverage


def feature_block(reference, labels, bodies, queries, query_bodies=None, *, seed=20260923):
    r, q = unit(reference), unit(queries)
    y = checked_labels(labels, len(r))
    order = np.argsort([normalize(t) for t in bodies], kind="stable")
    r, y = r[order], y[order]
    if r.shape[1] != q.shape[1]:
        raise ValueError("reference/query dimensions differ")
    label_null = np.random.default_rng(seed + 1).permutation(y)
    banks, diagnostics = {}, []
    original_assignments = {}
    for cls in (0, 1):
        original_assignments[cls] = spherical_partition(r[y == cls])
    for mode in MODES:
        target = label_null if mode == "label_null" else y
        locations, coverages = [], []
        for cls in (0, 1):
            x = r[target == cls]
            assign = (
                spherical_partition(x) if mode == "label_null" else original_assignments[cls].copy()
            )
            if mode == "random_partition":
                assign = np.random.default_rng(seed + 10 + cls).permutation(assign)
            elif mode == "single_center":
                assign = np.zeros(len(x), dtype=int)
            model = prototypes(x, assign)
            a, b = class_features(q, model)
            locations.append(a)
            coverages.append(b)
            diagnostics.append(
                {
                    "mode": mode,
                    "class": cls,
                    "reference_rows": len(x),
                    "prototype_count": len(model[0]),
                    "smallest_mode_fraction": float(model[1].min()),
                    "largest_mode_fraction": float(model[1].max()),
                    "mean_radius": float(model[2].mean()),
                    "outside_all_modes_fraction": float(np.mean(b[:, 4] == 0)),
                    "query_used_for_clustering": False,
                }
            )
        banks[mode] = np.column_stack(
            [
                *locations,
                locations[1] - locations[0],
                *coverages,
                coverages[1] - coverages[0],
            ]
        )
    return banks, diagnostics


def cross_fitted(reference, labels, bodies, rules, queries, query_bodies, *, seed=20260923):
    r, y, texts, q, qt, groups = validate_inputs(
        reference, labels, bodies, rules, queries, query_bodies
    )
    train = {m: np.empty((len(r), 36)) for m in MODES}
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
        raise ValueError("nonfinite multi-prototype feature bank")
    return banks, diagnostics
