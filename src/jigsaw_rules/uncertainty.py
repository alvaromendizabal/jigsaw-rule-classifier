"""Paired comment-group bootstrap intervals conditional on the observed rules."""

import numpy as np
from sklearn.metrics import roc_auc_score


def paired_auc_interval(y, baseline, candidate, rules, groups, *, draws=500, seed=2025):
    y, baseline, candidate, rules, groups = map(np.asarray, (y, baseline, candidate, rules, groups))
    if y.ndim != 1 or any(x.shape != y.shape for x in [baseline, candidate, rules, groups]):
        raise ValueError("Bootstrap inputs must be aligned 1-D arrays")
    if (
        not len(y)
        or draws < 10
        or not np.isfinite(baseline).all()
        or not np.isfinite(candidate).all()
    ):
        raise ValueError("Bootstrap requires finite scores and at least ten draws")
    labels = sorted(set(rules))
    _, inverse = np.unique(groups, return_inverse=True)
    count = inverse.max() + 1
    rng = np.random.default_rng(seed)

    def difference(weights):
        differences = []
        for rule in labels:
            mask = (rules == rule) & (weights > 0)
            if len(set(y[mask])) != 2:
                return None
            differences.append(
                roc_auc_score(y[mask], candidate[mask], sample_weight=weights[mask])
                - roc_auc_score(y[mask], baseline[mask], sample_weight=weights[mask])
            )
        return float(np.mean(differences))

    observed = difference(np.ones(len(y)))
    if observed is None:
        raise ValueError("Every rule must contain both classes")
    values = []
    attempts = 0
    while len(values) < draws and attempts < draws * 10:
        attempts += 1
        weights = np.bincount(rng.integers(count, size=count), minlength=count)[inverse]
        value = difference(weights)
        if value is not None:
            values.append(value)
    if len(values) < draws:
        raise ValueError("Too few mixed-class group bootstrap samples")
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "observed_delta": observed,
        "ci_lower": float(low),
        "ci_upper": float(high),
        "confidence_level": 0.95,
        "draws": draws,
        "attempts": attempts,
        "seed": seed,
        "resampling_unit": "normalized comment body, shared across rules",
        "scope": (
            "Conditional on these rules and fixed OOF predictions; "
            "not uncertainty across unseen policies."
        ),
    }
